# Verfahrensvergleich: welches Verfahren, gemessen woran? – Streamlit-Demo

Sechzehntes und letztes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es vergleicht die Verfahren der Stücke davor auf derselben Welt: eine lineare Entfaltung (n = 48) mit Gauß-Unschärfe (w = 0,03) und weißem Rauschen, gebaut auf [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo), [tsvd-demo](https://github.com/sebastian-hanisch/tsvd-demo), [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo), [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo), [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo), [plug-and-play-demo](https://github.com/sebastian-hanisch/plug-and-play-demo) und [diffusions-prior-demo](https://github.com/sebastian-hanisch/diffusions-prior-demo). Die Wahrheit ist bekannt, weil die Signale selbst die Wahrheit sind; sie dient hier nur als Maßstab.

## Kernfrage

**Es gibt kein bestes Verfahren, nur eines, dessen Annahme zur Welt passt, und ob sie passt, sagt weder der Datenrest noch die Abdeckung, sondern nur der Vergleich mit der Wahrheit, die man in der Praxis nicht hat.** Wer gewinnt wo? Was kostet die ehrliche Parameterwahl gegenüber dem Orakel? Unterscheidet der Datenrest die Verfahren? Löscht ein Verfahren Merkmale oder erfindet es welche? Wann sind Unsicherheitsbänder kalibriert? Erkennt die Evidenz die Welt?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Gitter, Operator, Rauschen, SplitMix64):

- **Daten:** b = A x + e mit der Gauß-Faltungsmatrix A (n = 48, w = 0,03), e ~ N(0, σ² I), σ je Wahrheit = Rauschstufe (1 / 10 %) mal Effektivwert der exakten Daten.
- **Sechs Wahrheitsfamilien** (feste SplitMix64-Ströme): Spitzen (2 bis 4), Blöcke (2 bis 4), Dreieckswellen, Phantom (zwei Hügel und ein Kasten), GP-Ziehungen (Prior aus Stück 8, exponentielle Korrelation, τ = 0,5, ℓ = 0,12) und die Blockzahl-Mischung aus Stück 15 (ein, zwei oder drei Blöcke, Masse 8).
- **Acht Verfahren:** Tikhonov (SVD, 41 Werte von λ), abgeschnittene SVD, CGLS mit Abbruch, L1 (FISTA, 150 Schritte je λ, 25 Werte, warm gestartet), Total Variation (ADMM, 400 Schritte, 25 Werte), Plug-and-Play mit Haar-Weichschwelle (orthonormale Haar-Matrix, 4 Ebenen, ohne Zyklenverschiebung), GP-Posterior-Mittel und Posterior-Mittel der Blockzahl-Mischung (exakt).
- **Parameterregeln:** **Orakel** (bester Parameter je Wahrheit), **Diskrepanzprinzip** (der am stärksten regularisierte Kandidat mit Rest ≤ σ √n), **GCV** (Tikhonov und TSVD), **Evidenz** (GP: τ per Maximum der Evidenz); die Mischung hat keinen freien Parameter.
- **Kennzahlen:** Datenrest |A x̂ − b| / (σ √n); **stumme Richtungen** (Richtung i der SVD, wenn sᵢ |vᵢᵀ x| < σ), Energieanteil des Schätzers dort und Kosinus zur Wahrheit; **Abschnittszahl** (zusammenhängende Stücke über 50 % des Maximums); **Abdeckung** des 90-%-Bands (5 bis 95 %) und relative Breite; **Evidenz** von GP, weißem Prior und Mischung.
- **Bänder:** GP-Posterior (τ per Evidenz), Mischungs-Posterior (Perzentile von 400 exakten Ziehungen), **konformal** (Tikhonov mit Diskrepanz, Halbbreite je Koordinate aus dem ⌈(m + 1) · 0,9⌉-kleinsten absoluten Fehler auf m = 60 Ziehungen derselben Familie).

Alles ist erfunden; die Wahrheit sind die simulierten Signale.

## Methodik

- **Verdikt:** das gewählte Verfahren gegen den ehrlichen Sieger (höchstens 1,1-faches: „gehört zu den besten“; ab 2-fachem: „die Annahme passt nicht“), dazu die Spanne von Datenrest und Fehler über alle Verfahren, Energie in stummen Richtungen, Abschnittszahl (löscht oder erfindet), ob das Band kalibriert ist (Abdeckung 85 bis 95 %) und welcher Prior die größte Evidenz hat.
- **Experimente** laufen über feste Zufallsströme (Seeds ab 1000), je mit eigenem Rechenknopf (10 bis 30 Sekunden); die Einstellung oben rechnet sofort (2 bis 5 Sekunden). **Kleinere Zahlen als in der Vorab-Messreihe:** 40 statt 100 Wahrheiten je Zelle, 150 statt 300 FISTA-Schritte je λ; die Zahlen sind Richtwerte, keine Konfidenzintervalle.
- **Gleichstände und letzte Stellen** werden nicht festgelegt: Bänder und Ungleichungen (bei Familien mit eng beieinanderliegenden Siegern, Wellen, Phantom und GP-Ziehung, nur die Klasse des Siegers).

## Befunde (gemessen, keine Behauptungen)

Median über 40 Wahrheiten je Zelle; ehrliche Parameterwahl, wenn nicht anders genannt. Jede Aussage steht als Band oder Ungleichung in `tests/test_claims.py` oder `tests/test_presets.py`.

| Frage | Befund | Test |
|---|---|---|
| Wer gewinnt wo (Experiment 1, Rauschen 1 % / 10 %)? | **Spitzen:** L1 **0,8 / 8,1 %**, alle anderen 34 bis 77 / 70 bis 82 %. **Blöcke:** TV **2,9 / 20,5 %** (Tikhonov 17,6 / 24,0 %). **Wellen:** Tikhonov 9,7 % bzw. CGLS 16,0 %; **Phantom:** Tikhonov 9,4 % bzw. CGLS 15,8 %; **GP-Ziehung:** GP 20,6 / 29,5 %. **Blockzahl:** die Mischung **12,8 / 19,9 %** (bestes anderes Verfahren mit Orakel-Parameter: TV 16,7 / 30,8 %). Sieger: L1, TV, Tikhonov oder CGLS, GP, Mischung. | `test_experiment_1_every_family_has_its_own_winner_class` |
| Gilt der Sieger auch in fremden Welten? | **Nein:** L1 liegt außer auf Spitzen bei **45 bis 86 %**; die Mischung auf Wellen bei **17,1 / 32,6 %** (Tikhonov 9,7 %, CGLS 16,0 %), auf dem Phantom bei 39,9 % (CGLS 15,8 %), auf Spitzen bei 59 / 76 %. | `test_experiment_1_the_mixture_wins_its_own_world_…` |
| Was kostet die ehrliche Parameterwahl? | Bis zum **1,9-Fachen**: PnP auf Spitzen 33,9 gegen 18,3 % mit Orakel (1 %), TV auf Blöcken 2,9 gegen 1,6 %; bei Tikhonov höchstens das 1,3-Fache. Das Orakel ist nie schlechter und steht als Messlatte da. | `test_experiment_1_the_oracle_is_never_worse_…` |
| Unterscheidet der Datenrest die Verfahren? | **Nein:** der Rest liegt bei allen Verfahren, Familien und Rauschstufen zwischen **0,5 und 1,25 σ**; bei Spitzen (10 %) zwischen 0,75 und 0,96 σ, die Fehler zwischen **8 und 82 %**. | `test_experiment_1_the_data_residual_…` |
| Wo steckt Erfindung (Experiment 2, Rauschen 10 %)? | In den stummen Richtungen (25 bis 35 von 48 je Familie) legt **L1 außer auf Spitzen 41 bis 49 %** der Energie ab (Wahrheit 1,4 bis 7,6 %; Kosinus −0,02 bis 0,15), auf Spitzen stimmt es (Kosinus 1,00); Tikhonov 0,6 bis 1,1 %, CGLS 0,1 bis 0,3 %, GP 0,8 bis 1,3 %, TV 3 bis 6 %, PnP 4 bis 13 %, Mischung 7 bis 12 % (Kosinus bei Blockzahl **0,88**, sonst unter 0,1). | `test_experiment_2_l1_puts_energy_…` |
| Löschen oder erfinden? | **Drei Blöcke, 10 %:** Tikhonov, TSVD, CGLS, PnP **94 %**, TV **100 %**, GP 86 % der Läufe mit zu wenigen Abschnitten (gelöscht); L1 mehr in 33 %; die Mischung hat so viele Abschnitte wie die Wahrheit in **92 %**, eine Probe in 81 %. **Ein Block, 10 %:** L1 zu viele in **90 %**, eine Probe aus dem exakten Posterior in **15 %** (TV 0 %). | `test_experiment_2_smooth_methods_delete_features_…` |
| Wann ist ein Band kalibriert (Experiment 3, 90-%-Band, Rauschen 10 %)? | **GP (τ = 0,5):** GP-Ziehung **0,91**, Blöcke 0,90, Spitzen **0,80**, Wellen **0,98**, Phantom **0,99** (zu breit). **Mischung:** Blockzahl **0,90** (Breite 0,65 gegen 0,87 beim GP), sonst **0,64 bis 0,87**. **Konformal** (auf die Familie kalibriert): **0,89 bis 0,91**; **auf Wellen kalibriert:** Spitzen **0,76**, Blöcke 0,77, GP-Ziehung **0,69**, Blockzahl **0,70**, Phantom 0,93. | `test_experiment_3_…` |
| Erkennt die Evidenz die Welt (Experiment 4)? | Der zur Familie passende Prior (Spitzen: weißer Prior, Blockzahl: Mischung, sonst GP) hat in **90 bis 100 %** der Läufe die größte Evidenz. Log-Evidenz Mischung minus GP: **+22,5 / +12,7** bei Blockzahl, **−78 bis −14** sonst. | `test_experiment_4_…` |
| Einzelstichproben (Presets, Seed 5, 40 Wahrheiten) | **Standard** (Blöcke, TV, 10 %): TV gewinnt mit 21,6 %, das GP-Band deckt 89,8 % ab. **Spitzen:** L1 8,1 %, das GP-Band 82 %. **Falsche Welt** (Mischung auf Wellen): 2,4-faches des Siegers, Band 0,83. **Blockzahl:** Mischung 17,4 %, Band 0,90 bei Breite 0,61. | `tests/test_presets.py` |
| Mini-Instanz von Hand | n = 3 (A symmetrisch, Eigenwerte 1 / 0,5 / 0,1), b = A x = (1,3, 1,4, 1,3), σ = 0,1: TSVD k = 1 gibt (1,3333; 1,3333; 1,3333) mit Rest 0,08165 und GCV 0,005 (k = 2: 0,02); Tikhonov λ = 0,1 gibt (1,1818; 1,2727; 1,1818) mit Rest 0,2227 und GCV 0,0900; CGLS k = 1 (1,33; 1,34; 1,33); stumme Richtungen 2 und 3, Energieanteil dort 1/9. | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Der konformale Fehler wird mit Tikhonov-Diskrepanz klein, mit GCV groß:** in der Vorab-Messreihe (GCV, 60 Ziehungen, interpolierte Quantile) lagen die Breiten bei 1,0 bis 6,3 mal dem Effektivwert, im Bau (Diskrepanz, Ordnungsstatistik) bei 0,55 bis 2,4; die Abdeckung für die Kalibrierungsfamilie bleibt bei 0,89 bis 0,91. Der Einbruch bei Familienwechsel (auf Wellen kalibriert) ist bei Spitzen kleiner (0,76 statt 0,60), bei der GP-Ziehung größer (0,69 statt 0,82) und bei Blockzahl gleich (0,70 gegen 0,68); nur beim Phantom bleibt die Abdeckung (0,93).
- **Die Evidenz ist kein Freibrief:** 90 bis 100 % Treffer gelten nur unter den drei angebotenen Priors; für Blöcke, Wellen und Phantom ist keiner richtig, und das beste Verfahren (TV bei Blöcken) hat mit keinem Prior zu tun.
- **Die Tikhonov-Handwerte aus Stück 14 gelten für λ = 0,01, nicht 0,1:** (1,153465; 1,653465; 1,153465) ist die Lösung bei λ = 0,01; bei λ = 0,1 ist sie (1,1818; 1,2727; 1,1818).
- **L1 mit Diskrepanzprinzip ist nicht L1 mit bestem λ:** auf glatten Welten wählt das Prinzip ein zu großes λ (Fehler 79 bis 86 % bei 10 %, 45 bis 65 % bei 1 %), mit Orakel-λ liegen 63 bis 73 % beziehungsweise 35 bis 53 %.
- **Der Zwillingstest** (zwei Signale mit gleichen Daten) wurde gemessen und verworfen: der Abstand der Schätzer zu den Zwillingen ist 4- bis 30-mal so groß wie der Zwillingsabstand; der Merkmalstest ersetzt ihn.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Wahrheit ist bekannt** | In der Praxis nicht: Orakel, Abdeckung, Abschnittszahl und Energie in stummen Richtungen brauchen sie. Ohne Wahrheit bleiben Datenrest (unterscheidet nichts) und Evidenz (nur unter den Kandidaten). | – |
| **Der Vergleich ist fair** | Die Einstellungen sind gewählt (FISTA mit 150 Schritten je λ, ADMM mit 400, Haar ohne Zyklenverschiebung, Diskrepanzprinzip mit τ = 1); andere Einstellungen ändern Zahlen, und 40 Wahrheiten geben Richtwerte. | – |
| **Die Welt ist eine der sechs** | n = 48, eine Dimension, Unschärfe 0,03, ein Operator; keine Bilder, kein trainiertes Netz, kein MCMC, keine Ensemble-Kalman-Inversion, kein Diffusionssampler mit TMM oder DPS (Stück 8, 10, 14, 15). | – |
| **Halluzination ist messbar** | Hier nur als zwei Maße in dieser Welt (Energie in stummen Richtungen, Abschnittszahl); keine Aussage über Bilder oder Anwender. | – |
| **Konformal löst die Kalibrierung** | Nur wenn Kalibrierung und Anwendung austauschbar sind; bei Familienwechsel bricht die Abdeckung, mit 60 Ziehungen und Koordinatenquantilen sind die Bänder breit. | – |
| **Die Evidenz entscheidet** | Nur unter den angebotenen Priors; für Blöcke, Wellen und Phantom ist keiner der drei richtig. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo), [tsvd-demo](https://github.com/sebastian-hanisch/tsvd-demo), [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo), [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo): Stück 2, 3, 5, 6; die Verfahren dieses Vergleichs, hier mit ehrlicher Parameterwahl und gegen das Orakel.
- [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo): Stück 4; GCV und Diskrepanzprinzip, hier an sechs Familien statt einer.
- [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo) und [posterior-sampling-demo](https://github.com/sebastian-hanisch/posterior-sampling-demo): Stück 7 und 8; der GP-Posterior und seine Kalibrierung (dort: Abdeckung 64 % auf dem Phantom), hier mit Wahrheitswechsel.
- [plug-and-play-demo](https://github.com/sebastian-hanisch/plug-and-play-demo), [entrollte-netze-demo](https://github.com/sebastian-hanisch/entrollte-netze-demo), [diffusions-prior-demo](https://github.com/sebastian-hanisch/diffusions-prior-demo): Stück 13 bis 15; der Denoiser als Prior (hier PnP-Haar), das gelernte Netz (hier nicht nachgebaut) und die Mischung mit exaktem Posterior (hier das Verfahren und der Sampler).

## Tests

Mini-Instanz von Hand (TSVD, Tikhonov, GCV, CGLS, Diskrepanz-Regel, Haar-Matrix, L1, TV, GP, Mischung, stumme Richtungen, Abschnittszahl, Abdeckung, konformale Halbbreite), Orakel (Tikhonov gegen die Normalgleichung, GCV gegen die explizite Spur, CGLS gegen den Krylov-Raum, L1 gegen die exakte Lasso-Lösung, TV gegen die KKT-Bedingungen, GP-Evidenz gegen die dichte Formel, Mischung mit einer Komponente gegen das GP, konformale Garantie per Monte-Carlo und ihr Bruch bei Familienwechsel), Generator, Auswertung (jede Schwelle des Urteils auf der Kante), jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Experimentknöpfe, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Gitter, Operator) |
| `vfv_model.py` | Wahrheitsfamilien, acht Verfahren mit Parameterregeln, Mischung, GP, Kennzahlen, konformale Halbbreite |
| `vfv_evaluation.py` | Vergleich auf einer Einstellung, vier Experimente, Verdikte |
| `vfv_visualization.py`, `vfv_pdf_export.py` | Figuren, PDF |
| `vfv_constants.py`, `vfv_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Trainierte Netze, MCMC, Ensemble-Kalman-Inversion, Diffusionssampler mit Näherungen des Likelihood-Scores:** eigene Stücke (8, 10, 14, 15); hier nur die exakten Größen.
- **Weitere Verfahren (Elastic Net, nichtlokale Mittel, Bayes-Hierarchien) und weitere Parameterregeln (L-Kurve, SURE, Kreuzvalidierung):** acht Verfahren und drei ehrliche Regeln genügen für die Aussage.
- **Zwillingstest, Bildmaße (SSIM, Wahrnehmung):** gemessen und verworfen beziehungsweise nicht gebaut.
- **Konfidenzintervalle:** ein Lauf je Zelle, feste Seeds.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
