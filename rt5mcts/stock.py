"""Stock (available building blocks) lookup."""
from __future__ import annotations

from typing import Set

from .chem import canonicalize


class Stock:
    """A set of purchasable molecules, compared by RDKit canonical SMILES."""

    def __init__(self, smiles_iterable) -> None:
        self._set: Set[str] = set()
        for smi in smiles_iterable:
            canon = canonicalize(smi.strip())
            if canon is not None:
                self._set.add(canon)

    @classmethod
    def from_file(cls, path: str) -> "Stock":
        with open(path) as fh:
            return cls(line for line in fh if line.strip())

    def __contains__(self, smiles: str) -> bool:
        canon = canonicalize(smiles)
        return canon is not None and canon in self._set

    def __len__(self) -> int:
        return len(self._set)
