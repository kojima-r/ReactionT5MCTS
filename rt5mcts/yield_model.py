"""Cached wrapper around ``sagawa/ReactionT5v2-yield`` for step-yield prediction.

Mirrors ``test_reactiont5_yield.py``: a small regression head on top of the
ReactionT5 encoder/decoder predicts a reaction yield (percent).  We wrap it with
the same on-disk SQLite memoisation used for the one-step model so that each
unique reaction is scored at most once across the whole search / experiment.

``predict_yield(product, reactants)`` returns a yield in **[0, 1]**.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import List, Optional, Sequence

MODEL_NAME = "sagawa/ReactionT5v2-yield"


def reaction_string(product: str, reactants: Sequence[str],
                    reagents: Sequence[str] = ()) -> str:
    """Build the ReactionT5-yield input string (same format as the reference)."""
    react = ".".join(reactants)
    reag = ".".join(reagents)
    return f"REACTANT:{react}REAGENT:{reag}PRODUCT:{product}"


class ReactionT5Yield:
    """Cached yield predictor. Model is loaded lazily on first cache miss."""

    def __init__(self, cache_path: str = "cache/rt5_yield.sqlite",
                 device: str = "cpu", seed: int = 42,
                 num_threads: Optional[int] = None, max_length: int = 300) -> None:
        self.device = device
        self.seed = seed
        self.max_length = max_length
        self._num_threads = num_threads
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        self._cache = sqlite3.connect(cache_path, check_same_thread=False)
        self._cache.execute("PRAGMA journal_mode=WAL")
        self._cache.execute("PRAGMA busy_timeout=30000")
        self._cache.execute("PRAGMA synchronous=NORMAL")
        self._cache.execute(
            "CREATE TABLE IF NOT EXISTS yields ("
            "rxn TEXT PRIMARY KEY, value REAL NOT NULL)"
        )
        self._cache.commit()
        self.n_calls = 0
        self.n_hits = 0

    # -- lazy model load ----------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import logging
        import torch
        import torch.nn as nn
        from transformers import (AutoTokenizer, PreTrainedModel, T5Config,
                                   T5ForConditionalGeneration)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        class ReactionT5Yield2(PreTrainedModel):
            config_class = T5Config
            base_model_prefix = "model"

            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.model = T5ForConditionalGeneration(config)
                self.model.resize_token_embeddings(self.config.vocab_size)
                h = self.config.hidden_size
                self.fc1 = nn.Linear(h, h // 2)
                self.fc2 = nn.Linear(h, h // 2)
                self.fc3 = nn.Linear(h // 2 * 2, h)
                self.fc4 = nn.Linear(h, h)
                self.fc5 = nn.Linear(h, 1)
                self.post_init()
                self.all_tied_weights_keys = {}

            def forward(self, inputs):
                enc = self.model.encoder(**inputs)
                enc_h = enc[0]
                dec = self.model.decoder(
                    input_ids=torch.full(
                        (inputs["input_ids"].size(0), 1),
                        self.config.decoder_start_token_id,
                        dtype=torch.long, device=inputs["input_ids"].device),
                    encoder_hidden_states=enc_h,
                )
                h = self.config.hidden_size
                o1 = self.fc1(dec[0].view(-1, h))
                o2 = self.fc2(enc_h[:, 0, :].view(-1, h))
                o = self.fc3(torch.hstack((o1, o2)))
                o = self.fc5(self.fc4(o))
                return o * 100

        torch.manual_seed(self.seed)
        if self._num_threads:
            torch.set_num_threads(self._num_threads)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = ReactionT5Yield2.from_pretrained(MODEL_NAME).to(self.device).eval()

    # -- cache --------------------------------------------------------------
    def _get(self, rxn: str) -> Optional[float]:
        row = self._cache.execute(
            "SELECT value FROM yields WHERE rxn=?", (rxn,)).fetchone()
        return float(row[0]) if row else None

    def _put(self, rxn: str, value: float) -> None:
        self._cache.execute(
            "INSERT OR REPLACE INTO yields (rxn, value) VALUES (?,?)", (rxn, value))
        self._cache.commit()

    # -- inference ----------------------------------------------------------
    def _run(self, rxn: str) -> float:
        import torch
        self._ensure_model()
        inp = {k: v.to(self.device) for k, v in
               self._tokenizer([rxn], return_tensors="pt",
                               truncation=True, max_length=self.max_length).items()}
        with torch.no_grad():
            pct = float(self._model(inp).item())
        return pct

    def predict_yield(self, product: str, reactants: Sequence[str],
                      reagents: Sequence[str] = ()) -> float:
        """Predicted yield in [0, 1] for one reaction step (cached)."""
        rxn = reaction_string(product, reactants, reagents)
        with self._lock:
            cached = self._get(rxn)
            if cached is not None:
                self.n_hits += 1
                return cached
            pct = self._run(rxn)
            self.n_calls += 1
            val = max(0.0, min(1.0, pct / 100.0))
            self._put(rxn, val)
            return val

    def stats(self) -> dict:
        return {"yield_calls": self.n_calls, "yield_hits": self.n_hits}

    def close(self) -> None:
        self._cache.close()
