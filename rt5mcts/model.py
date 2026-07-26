"""ReactionT5 one-step retrosynthesis model wrapper with a persistent cache.

The wrapper around ``sagawa/ReactionT5v2-retrosynthesis`` (see ``test_reactiont5.py``)
produces, for a given product SMILES, a ranked list of candidate reactant sets
together with a probability-like score derived from the beam-search sequence
scores.

Because a single beam-search call on CPU costs ~10-17 s, every prediction is
memoised in an on-disk SQLite cache keyed by the canonical product SMILES and
the beam width.  We always query the model with a *fixed* (large) beam width and
cache the full candidate list; the MCTS then slices the top-`expansion_width`
candidates it needs.  This guarantees the model is evaluated at most once per
unique molecule across the entire experiment (and across every hyper-parameter
configuration sharing the same cache file).
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .chem import canonicalize, split_fragments, unique_sorted

MODEL_NAME = "sagawa/ReactionT5v2-retrosynthesis"
DEFAULT_MAX_BEAMS = 10
DEFAULT_MAX_LENGTH = 200


@dataclass(frozen=True)
class Prediction:
    """A single candidate reaction produced by the one-step model."""

    reactants: Tuple[str, ...]  # canonical, de-duplicated, sorted
    prob: float                 # normalised probability among the returned beams
    raw_logprob: float          # length-normalised log-prob from beam search


class ReactionT5:
    """Cached one-step retrosynthesis predictor."""

    def __init__(
        self,
        cache_path: str,
        max_beams: int = DEFAULT_MAX_BEAMS,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str = "cpu",
        seed: int = 42,
        num_threads: Optional[int] = None,
    ) -> None:
        self.max_beams = max_beams
        self.max_length = max_length
        self.device = device
        self.seed = seed
        self._num_threads = num_threads
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        self._cache = sqlite3.connect(cache_path, check_same_thread=False)
        # WAL + busy timeout so several worker processes can share one cache file
        self._cache.execute("PRAGMA journal_mode=WAL")
        self._cache.execute("PRAGMA busy_timeout=30000")
        self._cache.execute("PRAGMA synchronous=NORMAL")
        self._cache.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            "smiles TEXT NOT NULL, beams INTEGER NOT NULL, payload TEXT NOT NULL, "
            "PRIMARY KEY (smiles, beams))"
        )
        self._cache.commit()
        self.n_model_calls = 0
        self.n_cache_hits = 0
        # per-target budget on NEW (cache-miss) model calls; None = unlimited
        self.call_budget: Optional[int] = None
        self._budget_used = 0

    def reset_budget(self, n: Optional[int]) -> None:
        """Reset the per-target budget of fresh model evaluations."""
        self.call_budget = n
        self._budget_used = 0

    # -- lazy model loading -------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        torch.manual_seed(self.seed)
        if self._num_threads:
            torch.set_num_threads(self._num_threads)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, return_tensors="pt")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self._model.to(self.device)
        self._model.eval()

    # -- cache --------------------------------------------------------------
    def _cache_get(self, smiles: str) -> Optional[List[dict]]:
        row = self._cache.execute(
            "SELECT payload FROM predictions WHERE smiles=? AND beams=?",
            (smiles, self.max_beams),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, smiles: str, payload: List[dict]) -> None:
        self._cache.execute(
            "INSERT OR REPLACE INTO predictions (smiles, beams, payload) VALUES (?,?,?)",
            (smiles, self.max_beams, json.dumps(payload)),
        )
        self._cache.commit()

    # -- inference ----------------------------------------------------------
    def _run_model(self, product: str) -> List[dict]:
        import torch

        self._ensure_model()
        inp = self._tokenizer(product, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inp,
                num_beams=self.max_beams,
                num_return_sequences=self.max_beams,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=self.max_length,
            )
        seqs = [
            self._tokenizer.decode(s, skip_special_tokens=True).replace(" ", "").rstrip(".")
            for s in out["sequences"]
        ]
        if getattr(out, "sequences_scores", None) is not None:
            logps = [float(x) for x in out.sequences_scores.tolist()]
        else:
            logps = [0.0] * len(seqs)

        payload: List[dict] = []
        for smi, logp in zip(seqs, logps):
            frags = split_fragments(smi)
            if not frags:
                continue  # unparsable model output
            frags = unique_sorted(frags)
            payload.append({"reactants": frags, "logprob": logp})
        return payload

    def predict_raw(self, product_smiles: str) -> List[dict]:
        """Return the cached/raw candidate payload for a product (canonicalised)."""
        canon = canonicalize(product_smiles)
        if canon is None:
            return []
        with self._lock:
            cached = self._cache_get(canon)
            if cached is not None:
                self.n_cache_hits += 1
                return cached
            if self.call_budget is not None and self._budget_used >= self.call_budget:
                # budget exhausted for this target: treat uncached molecules as
                # non-expandable (deterministic given fixed search order)
                return []
            self._budget_used += 1
            payload = self._run_model(canon)
            self.n_model_calls += 1
            self._cache_put(canon, payload)
            return payload

    def predict(self, product_smiles: str, top_k: int) -> List[Prediction]:
        """Return up to ``top_k`` candidate reactions as :class:`Prediction`.

        Candidates that reproduce the product unchanged (no-op reactions) are
        dropped, and probabilities are re-normalised over the surviving
        candidates (softmax over the length-normalised log-probs).
        """
        canon = canonicalize(product_smiles)
        raw = self.predict_raw(product_smiles)

        kept = []
        for cand in raw:
            reactants = tuple(cand["reactants"])
            # drop trivial "product -> product" predictions
            if len(reactants) == 1 and reactants[0] == canon:
                continue
            if canon is not None and canon in reactants:
                continue
            kept.append((reactants, float(cand["logprob"])))
            if len(kept) >= top_k:
                break

        if not kept:
            return []

        logps = [lp for _, lp in kept]
        m = max(logps)
        exps = [math.exp(lp - m) for lp in logps]
        z = sum(exps)
        return [
            Prediction(reactants=rs, prob=e / z, raw_logprob=lp)
            for (rs, lp), e in zip(kept, exps)
        ]

    def stats(self) -> dict:
        return {"model_calls": self.n_model_calls, "cache_hits": self.n_cache_hits}

    def close(self) -> None:
        self._cache.close()
