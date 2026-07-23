"""Small chemistry helpers built on RDKit.

All SMILES handled inside the search are *canonical* (RDKit canonical SMILES)
so that molecule identity is well defined for the stock lookup and the
prediction cache.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from rdkit import Chem
from rdkit import RDLogger

# RDKit is very noisy about unparsable SMILES coming out of the model; silence it.
RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=200_000)
def canonicalize(smiles: str) -> Optional[str]:
    """Return the RDKit canonical SMILES, or ``None`` if it cannot be parsed."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def split_fragments(smiles: str) -> List[str]:
    """Split a (possibly multi-component) SMILES into canonical fragments.

    Returns an empty list if *any* fragment is unparsable, which lets callers
    reject invalid model outputs cleanly.
    """
    frags: List[str] = []
    for part in smiles.split("."):
        part = part.strip()
        if not part:
            continue
        canon = canonicalize(part)
        if canon is None:
            return []
        frags.append(canon)
    return frags


def unique_sorted(frags: List[str]) -> List[str]:
    """De-duplicate fragments while keeping a deterministic (sorted) order."""
    return sorted(set(frags))
