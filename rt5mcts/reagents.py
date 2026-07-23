"""Common reagents / solvents / salts treated as always-available.

ReactionT5 was trained on full USPTO reactions and therefore emits reagents,
catalysts, solvents and counter-ions as part of the predicted reactant set.
The PaRoutes stock, by contrast, only contains the *building blocks* of the
reference routes.  Without augmenting the stock with these ubiquitous reagents,
otherwise-valid routes never "close" (a leaf such as tetrabutylammonium can
never be traced back to a building block).  Treating a curated set of common
reagents as purchasable is chemically justified and standard practice.
"""
from __future__ import annotations

from typing import List

from .chem import canonicalize

# A pragmatic list of ubiquitous reagents/solvents/bases/acids/catalysts/ions.
_RAW = [
    # water / simple inorganics / gases
    "O", "[H][H]", "O=O", "N#N", "O=C=O", "[C-]#[O+]", "N", "Cl", "Br", "I", "F",
    "[Cl-]", "[Br-]", "[I-]", "[F-]", "[Na+]", "[K+]", "[Li+]", "[Cs+]",
    "[OH-]", "[H-]", "[NH4+]",
    # acids
    "Cl", "O=S(=O)(O)O", "OC(=O)C(F)(F)F", "CC(=O)O", "O=C(O)O", "OP(=O)(O)O",
    "O=[N+]([O-])O", "OS(=O)(=O)C(F)(F)F", "Cc1ccc(S(=O)(=O)O)cc1",
    # bases
    "CCN(CC)CC", "CCN(C(C)C)C(C)C", "c1ccncc1", "Cn1ccnc1",
    "[Na+].[OH-]", "[K+].[OH-]", "O=C([O-])[O-].[K+].[K+]",
    "O=C([O-])[O-].[Cs+].[Cs+]", "O=C([O-])[O-].[Na+].[Na+]",
    "[Na+].CC(C)(C)[O-]", "[K+].CC(C)(C)[O-]", "CC(C)(C)[O-].[Na+]",
    "[Na+].[H-]", "[Li]CCCC", "C[Si](C)(C)[N-][Si](C)(C)C",
    # reducing / oxidising / hydride sources
    "[Al+3].[Li+].[H-].[H-].[H-].[H-]", "[BH4-].[Na+]", "[BH3-]C#N.[Na+]",
    "O=[Mn](=O)(=O)[O-].[K+]", "O=S(C)C", "[O-][Cl+3]([O-])([O-])[O-]",
    # coupling / activating reagents
    "CCOC(=O)/N=N/C(=O)OCC", "CC(C)OC(=O)/N=N/C(=O)OC(C)C",
    "O=C(O)/N=N/C(=O)O", "C(=NC1CCCCC1)=NC1CCCCC1",
    "CCN=C=NCCCN(C)C", "O=C(n1ccnc1)n1ccnc1",
    "CN(C)C(On1nnc2ccccc21)=[N+](C)C", "F[P-](F)(F)(F)(F)F",
    # protecting-group / acylating reagents
    "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "O=C(Cl)OCc1ccccc1", "CC(=O)OC(C)=O",
    "O=C(Cl)C(Cl)=O", "ClC(Cl)=O", "CS(=O)(=O)Cl", "Cc1ccc(S(=O)(=O)Cl)cc1",
    # phosphines / catalysts / ligands (as small proxies)
    "c1ccc(P(c2ccccc2)c2ccccc2)cc1", "[Pd]", "[Pt]", "[Ni]",
    "CC(=O)O[Pd]OC(C)=O",
    # ammonium / phase-transfer
    "CCCC[N+](CCCC)(CCCC)CCCC", "CCCC[N+](CCCC)(CCCC)CCCC.[F-]",
    "CCCC[N+](CCCC)(CCCC)CCCC.[Br-]",
    # common solvents
    "ClCCl", "ClC(Cl)Cl", "ClC(Cl)(Cl)Cl", "C1CCOC1", "CO", "CCO", "CC#N",
    "CN(C)C=O", "CC(C)=O", "CC(=O)N(C)C", "c1ccccc1", "Cc1ccccc1",
    "CCOCC", "CCOC(C)=O", "Clc1ccccc1", "O=S(=O)(C)C", "CN1CCCC1=O",
    "OCC", "CO", "C1CCCCC1", "CCCCCC",
    # misc small building reagents
    "C=O", "CI", "CBr", "CCl", "BrCBr", "N#CBr", "O=CO",
]


def reagent_smiles() -> List[str]:
    """Return canonicalised, de-duplicated reagent SMILES."""
    out = set()
    for smi in _RAW:
        canon = canonicalize(smi)
        if canon is not None:
            out.add(canon)
    return sorted(out)
