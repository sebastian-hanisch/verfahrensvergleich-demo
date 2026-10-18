"""Konstanten: Regler, Wahrheitsfamilien, Verfahren, Experimente, Farben und Presets."""

# Welt: die lineare Entfaltung aus Stück 1 bis 15 (n = 48 Gitterpunkte, Gauß-Unschärfe), Daten b = A x + e (das Modell stimmt).
N = 48
W = 0.03                                                         # Breite der Unschärfe (Anteil der Länge 1), fest
NOISE_CHOICES = (1e-2, 1e-1)                                     # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-1
FAMILIES = ("spikes", "blocks", "waves", "phantom", "gauss", "blockzahl")
FAMILY_LABELS = {"spikes": "Spitzen (2 bis 4)", "blocks": "Blöcke (2 bis 4)", "waves": "Dreieckswellen", "phantom": "Phantom (Hügel und Kasten)", "gauss": "GP-Ziehung (Prior aus Stück 8)",
                 "blockzahl": "Blockzahl-Mischung (Stück 15)"}
FAMILY_SHORT = {"spikes": "Spitzen", "blocks": "Blöcke", "waves": "Wellen", "phantom": "Phantom", "gauss": "GP-Ziehung", "blockzahl": "Blockzahl"}
METHODS = ("tikhonov", "tsvd", "cgls", "l1", "tv", "pnp", "gp", "mixture")
METHOD_LABELS = {"tikhonov": "Tikhonov (Stück 2)", "tsvd": "Abgeschnittene SVD (Stück 3)", "cgls": "CGLS mit Abbruch (Stück 5)", "l1": "L1, Weichschwelle (Stück 6)", "tv": "Total Variation (Stück 6)",
                 "pnp": "Plug-and-Play mit Haar-Weichschwelle (Stück 13)", "gp": "GP-Posterior-Mittel (Stück 7)", "mixture": "Blockzahl-Mischung, Posterior-Mittel (Stück 15)"}
METHOD_SHORT = {"tikhonov": "Tikhonov", "tsvd": "TSVD", "cgls": "CGLS", "l1": "L1", "tv": "TV", "pnp": "PnP-Haar", "gp": "GP-Mittel", "mixture": "Mischung"}
RULES = ("honest", "oracle")
RULE_LABELS = {"honest": "ehrlich (Diskrepanz, beim GP die Evidenz)", "oracle": "Orakel (bester Parameter, Wahrheit bekannt)"}
BANDS = ("gp", "mixture", "conformal")
BAND_LABELS = {"gp": "GP-Posterior (τ aus der Evidenz)", "mixture": "Mischungs-Posterior (Perzentile von Ziehungen)", "conformal": "konformal (Tikhonov, kalibriert auf dieselbe Familie)"}
COUNT_CHOICES = (20, 40, 100)
DEFAULT_COUNT = 40
SEED_MAX, DEFAULT_SEED = 999999, 5

# Wahrheitsfamilien: je Signal DRAW_WIDTH gleichverteilte Zahlen aus dem Strom (wie Stück 14)
DRAW_WIDTH = 16
SPIKE_COUNT, SPIKE_HEIGHT = (2, 3), (1.0, 1.5)
BLOCK_COUNT, BLOCK_WIDTH, BLOCK_HEIGHT = (2, 3), (3, 8), (0.3, 0.9)
WAVE_PERIODS, WAVE_RANGE = (3, 3), (0.1, 0.9)
PHANTOM_CENTER, PHANTOM_WIDTH, PHANTOM_HEIGHT = (0.2, 0.6), (0.03, 0.03), (0.5, 0.5)
PHANTOM_BOX_LEFT, PHANTOM_BOX_HEIGHT, PHANTOM_BOX_WIDTH = (0.1, 0.7), (0.3, 0.4), 0.12
PRIOR_TAU, PRIOR_ELL = 0.5, 0.12                                 # Gauß-Prior wie in Stück 8
MIX_TAU, MIX_ELL, MIX_MASS = 0.3, 0.12, 8.0                      # Blockzahl-Mischung wie in Stück 15
ONE_BLOCK, TWO_BLOCKS, THREE_BLOCKS = (20, 28), ((20, 23), (25, 28)), ((20, 22), (23, 25), (26, 28))
BLOCK_CELLS_MULTI = 6

# Verfahren
LAMBDA_COUNT, LAMBDA_SPAN = 41, 10                               # Tikhonov: lambda = s_max² · 10^(0 ... -10), 41 Werte
CGLS_ITERATIONS = 40
SOFT_LAMBDAS, SOFT_SPAN, SOFT_STEPS = 25, 4, 150                 # L1 und PnP: lambda relativ zu |A^T b|_inf, 10^(0 ... -4), FISTA-Schritte je lambda
TV_LAMBDAS, TV_SPAN, TV_START, TV_STEPS, TV_RHO = 25, 4.5, 0.5, 400, 1.0
HAAR_LEVELS = 4
DISCREPANCY_TAU = 1.0                                            # Diskrepanzprinzip: Rest <= tau sigma sqrt(n)
GP_TAU_GRID = (-2.0, 1.5, 40)                                    # tau = 10^(-2 ... 1,5), 40 Werte (Evidenz)
GP_FIXED_TAU = 0.5
CONFORMAL_CAL = 60                                               # Kalibrierungsmenge (Ziehungen derselben Familie)
LEVEL = 0.9                                                      # nominale Abdeckung (Band 5 bis 95 %)
Z90 = 1.6449
MIX_DRAWS = 400                                                  # Ziehungen für das Mischungsband
WAVES_CAL = "waves"                                              # Wahrheitswechsel: auf Wellen kalibriert
SEED_TRUTH, SEED_NOISE, SEED_CAL, SEED_DRAW = 1, 2, 3, 4         # Offsets zum Seed der Einstellung

# Verdikt-Schwellen
WIN_TOL = 1.10                                                   # gehört zu den besten: Fehler höchstens 1,1 mal der des Siegers
BAD_FACTOR = 2.0                                                 # mehr als doppelt so schlecht wie der Sieger
SEG_FRAC = 0.5                                                   # Abschnitt: Wert über 50 % des Maximums
COVER_LO, COVER_HI = 0.85, 0.95                                  # Abdeckung gilt als kalibriert, wenn sie in diesem Bereich liegt

# Experimente: feste Ströme (Seeds ab SEED_BASE), kleinere Zahlen als in der Messreihe (40 statt 100 Wahrheiten, 150 statt 300 Schritte)
SEED_BASE = 1000
EXP_COUNT = 40
EXP_CAL = 60
EXP_MIX_DRAWS = 200
FEATURE_NOISES = (1e-2, 1e-1)

COLOR_TRUTH = "#2F6B65"
COLOR_EST = "#4c78a8"
COLOR_ORACLE = "#D68A2E"
COLOR_DATA = "#8a8a8a"
COLOR_BAND = "rgba(76,120,168,0.22)"
COLOR_MIX = "#c0392b"
METHOD_COLORS = {"tikhonov": "#4c78a8", "tsvd": "#8fb3d9", "cgls": "#1f4e79", "l1": "#c0392b", "tv": "#D68A2E", "pnp": "#8e5fbf", "gp": "#c2549f", "mixture": "#222222"}
SHOW_RANGE = {"spikes": (-0.8, 3.4), "blocks": (-0.5, 2.6), "waves": (-0.5, 2.4), "phantom": (-0.5, 2.0), "gauss": (-1.8, 1.8), "blockzahl": (-1.0, 2.4)}   # feste Achse der Rekonstruktion je Familie
TABLE_ROWS = 40

_BASE = {"family": "blocks", "noise": DEFAULT_NOISE, "method": "tv", "rule": "honest", "band": "gp", "count": DEFAULT_COUNT, "seed": DEFAULT_SEED}
PRESETS = {
    "Standard": dict(_BASE),
    "Spitzen": dict(_BASE, family="spikes", method="l1"),
    "Wellen": dict(_BASE, family="waves", method="cgls"),
    "Blockzahl": dict(_BASE, family="blockzahl", method="mixture", band="mixture"),
    "Falsche Welt": dict(_BASE, family="waves", method="mixture", band="mixture"),
    "Wenig Rauschen": dict(_BASE, noise=1e-2),
    "Konformal": dict(_BASE, family="phantom", method="tikhonov", band="conformal"),
}
PRESET_HELP = {
    "Standard": "Blöcke, Rauschen 10 %: TV gewinnt, das GP-Band ist kalibriert.",
    "Spitzen": "Spitzen: L1 gewinnt mit großem Abstand, das GP-Band ist zu schmal.",
    "Wellen": "Wellen: CGLS gewinnt; alle Verfahren treffen die Daten gleich gut, ihre Fehler unterscheiden sich stark.",
    "Blockzahl": "Die eigene Welt der Mischung: sie gewinnt, und ihr Band ist kalibriert und schmal.",
    "Falsche Welt": "Dieselbe Mischung auf Wellen: mehr als doppelt so schlecht wie der Sieger, das Band ist nicht kalibriert.",
    "Wenig Rauschen": "Blöcke bei Rauschen 1 %: alle Fehler sinken, der Sieger bleibt TV.",
    "Konformal": "Phantom mit konformalem Band: die Abdeckung hält, solange die Wahrheit aus der Familie der Kalibrierung kommt.",
}
