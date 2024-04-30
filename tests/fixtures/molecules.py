# Copyright © 2024 HQS Quantum Simulations GmbH. All Rights Reserved.
from pyscf.gto import M, Mole


def create_mol(name: str, **kwargs) -> Mole:
    if name not in MOL:
        raise ValueError("Unknown molecule name.")
    return M(**{**MOL[name], **kwargs})


MOL = {
    "helium_atom": {
        "atom": "He 0 0 0",
        "basis": "6-31G",
        "spin": 0,
    },
    "lithium_atom": {
        "atom": "Li 0 0 0",
        "basis": "6-31G",
        "spin": 1,
    },
    "beryllium_atom": {
        "atom": "Be 0 0 0",
        "basis": "6-31G",
        "spin": 0,
    },
    "carbon_atom": {
        "atom": "C 0 0 0",
        "basis": "6-31G",
        "spin": 2,
    },
    "nitrogen_atom": {
        "atom": "N 0 0 0",
        "basis": "6-31G",
        "spin": 3,
    },
    "nitrogen": {
        "atom": """
            N  0.00   0.00  -0.55
            N  0.00   0.00   0.55
            """,
        "basis": "def2-SVP",
        "charge": 0,
        "spin": 0,
    },
    "oxygen": {
        "atom": """
            O   0.00   0.00  -0.60
            O   0.00   0.00   0.60
        """,
        "basis": "def2-SVP",
        "spin": 2,
        "charge": 0,
    },
    "OH_radical": {
        "atom": """
            O   0.00   0.00   0.00
            H   0.00   0.00   0.97
            """,
        "basis": "def2-SVP",
        "charge": 0,
        "spin": 1,
    },
    "nitric_oxide": {
        "atom": """
            N   0.00   0.00   0.58
            O   0.00   0.00  -0.58
        """,
        "basis": "def2-SVP",
        "charge": 0,
        "spin": 1,
    },
    "cyanide": {
        "atom": """
            C   0.00   0.00  -0.56
            N   0.00   0.00   0.56
        """,
        "basis": "def2-SVP",
        "charge": 0,
        "spin": 1,
    },
    "formaldehyde": {
        "atom": """
            C 0.000  0.000 -0.533
            O 0.000  0.000  0.680
            H 0.000 -0.937 -1.118
            H 0.000  0.937 -1.118
            """,
        "basis": "minao",
        "charge": 0,
        "spin": 0,
        "symmetry": False,
    },
    "formaldehyde_2": {
        "atom": """
            C     -0.012173    0.001963    0.000208
            O      1.196662   -0.193004   -0.020430
            H     -0.444033    1.015492    0.000371
            H     -0.740456   -0.824451    0.019851
            """,
        "basis": "def2-SVP",
        "charge": 0,
        "spin": 0,
    },
    "water": {
        "atom": """
            O  0.00   0.00   0.00
            H  0.00   0.96   0.00
            H  0.93  -0.24   0.00
            """,
        "basis": "def2-SVP",
        "charge": 0,
        "spin": 0,
    },
    "water_2": {
        "atom": """
            O   0.000   0.000   0.000
            H   0.000   0.757  -0.587
            H   0.000  -0.757  -0.587
            """,
        "basis": "def2-SVP",
        "charge": 0,
        "spin": 0,
    },
    "ammonia": {
        "atom": """
            N      0.000000000      0.000000000      0.000000000
            H      0.942775821      0.000000000     -0.333427276
            H     -0.471387910      0.816467811     -0.333427276
            H     -0.471387910     -0.816467811     -0.333427276
            """,
        "basis": "def2-SVP",
        "charge": 0,
        "spin": 0,
    },
    "allyl": {
        "atom": """
            C  0.00  0.00  0.00
            C -1.26 -0.66  0.00
            C  1.26 -0.66  0.00
            H  0.00  1.07  0.00
            H -2.18 -0.11  0.00
            H  2.18 -0.11  0.00
            H -1.26 -1.73  0.00
            H  1.26 -1.73  0.00
            """,
        "basis": "def2-SVP",
        "spin": 1,
    },
    "benzyl_radical": {
        "atom": """
            H      2.895599   -0.900733   -0.214931
            C      2.344429    0.027195   -0.130588
            H      2.879949    0.967729   -0.106778
            C      0.892001    0.010347   -0.049686
            C      0.176275    1.206532    0.059331
            C     -1.218501    1.190220    0.137015
            C     -1.905336   -0.022102    0.106131
            C     -1.198329   -1.218255   -0.002393
            C      0.196450   -1.202208   -0.080093
            H      0.695533    2.161337    0.084877
            H     -1.767617    2.124245    0.221591
            H     -2.990363   -0.034688    0.166569
            H     -1.731692   -2.164837   -0.026673
            H      0.731601   -2.144782   -0.164371
        """,
        "basis": "6-31G**",
        "charge": 0,
        "spin": 1,
    },
    "ethene": {
        "atom": """
            C       -0.669499940      0.000000000      0.000000000
            C        0.669499940      0.000000000      0.000000000
            H        1.231924190     -0.928841844      0.000000000
            H        1.231924190      0.928841844      0.000000000
            H       -1.231924190      0.928841844      0.000000000
            H       -1.231924190     -0.928841844      0.000000000
        """,
        "basis": "def2-SVP",
        "charge": 0,
        "spin": 0,
    },
}
