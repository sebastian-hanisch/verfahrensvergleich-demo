"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 14 (symmetrisch, Zeilensumme 1, Eigenwerte 1, 0,5, 0,1 zu den Vektoren (1,1,1), (1,0,-1), (1,-2,1)), Wahrheit x = (1, 2, 1) (|x|² = 6), Daten b = A x = (1,3, 1,4, 1,3),
    sigma = 0,1 (sigma sqrt(n) = 0,1732). Handrechnung mit beta = U^T b = (4/sqrt(3) = 2,3094; 0; -0,2/sqrt(6) = -0,08165):
    - TSVD k = 1: x = (1,3333; 1,3333; 1,3333), Rest 0,08165 (k = 2 gleich), k = 3: x = (1, 2, 1), Rest 0; GCV n r² / (n - k)²: k = 1: 0,005, k = 2: 0,02, k = 3: unendlich;
    - Tikhonov lambda = 0,1: Filter f = s² / (s² + lambda) = (0,9091; 0,7143; 0,0909), x = (1,1818; 1,2727; 1,1818), Rest 0,2227, GCV 0,0900; lambda = 0,01: x = (1,153465; 1,653465; 1,153465);
    - CGLS k = 1: A^T b = (1,33; 1,34; 1,33), Schrittweite 1 (|s|² = 5,3334, |A s|² = 5,3333): x_1 = (1,33; 1,34; 1,33);
    - stumme Richtungen für x = (1, 2, 1): Koeffizienten (2,3094; 0; -0,8165), s |c| = (2,3094; 0; 0,0816): Richtung 2 und 3 stumm, Energieanteil dort 0,6667 / 6 = 0,1111;
    - Datenrest von x_hat = 0: |b| / (sigma sqrt(n)) = 2,3108 / 0,1732 = 13,34."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([[1.0, 2.0, 1.0]])
    return {"a": a, "x": x, "b": x @ a.T, "sig": np.array([0.1])}
