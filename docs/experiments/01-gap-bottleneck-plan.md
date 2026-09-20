# Experiment-Plan 01 — Gap-Regression-Engpass

**Datum**: 2026-05-08
**Ausgangs-Branch**: `feature/gnn-optimization`
**Aktueller Checkpoint**: `gnn_models/gnn_latentdim8_batchsize1024_20260429-223735.pth`
**Constraint**: shape-agnostisch — keine Features, die nur für Boxen gelten. Repair optimiert immer nur über `z` und `pos`.

## Diagnose (zur Erinnerung)

| Label | MAE | Threshold | MAE/Thresh | Threshold-Agreement |
|---|---|---|---|---|
| under_surface | 0.091 mm | 1.000 mm | 9.1 % | 99.49 % |
| thickness | 0.614 mm | 22.000 mm | 2.8 % | 98.48 % |
| overlap | 1.190 mm | 10.000 mm | 11.9 % | 96.12 % |
| **planned_gap** | **0.398 mm** | **1.000 mm** | **39.8 %** | **94.24 %** |

Bucket `|gap| ∈ [0,3) mm` (n = 351 994):
- slope = 0.80, intercept = −0.249 mm
- pred unterschätzt true Magnitude und ist nach unten verschoben
- Bereich, in dem die Feasibility-Entscheidung lebt

Quelle: `python src/label_analysis/analyze_label_generalization.py`, vollständige Logs in `results/label_analysis/`.

## Was bereits probiert wurde

| Commit | Inhalt |
|---|---|
| `87e1017` `2e3995f` | 4-Klassen-Mode-Refactor, gap_low/gap_high → unified gap |
| `8c8afff` | Boundary- und magnitude-aware Gewichtung für gap-Regression |
| `374f3b1` | Gap-Threshold von 0.003 m auf 0.001 m (zweiseitig) verschärft |
| `43881a1` | Splitting Gap in `gap_mag` + `gap_sign`, separate Heads |
| `9fd2a95` `f4ec644` `f5d82e6` | Gap-Loss-Reweighting / Huber-Delta-Tuning |
| `14f93d1` | `THRESH_GAP_OPT = 0.7 × THRESH_GAP` (konservativeres Repair-Ziel) |

Mündlich von Fabio bestätigt:
- ursprünglich einfaches MLP für planned_gap (vor mag/sign-Split)
- threshold-relative Parametrierung wurde versucht, "hat nicht geklappt", möglicher Bug
- skeptisch gegen sign-aware Hinge im Repair (Begründung: Sign-Logit unter 1 mm rauschdominiert)

## Hypothesen-Katalog

### H1 — Shape-agnostisches Bounding-Box-Extent als Node-Feature (REVIDIERT)

**Idee**: Knoten-Features um die **shape-agnostische Bounding-Box-Extent** in jeder Achse erweitern: `[extent_x, extent_y, extent_z]` — die maximale Ausdehnung der Surface in jeder Achse, gemessen relativ zum Shape-Center. Für Boxen entspricht das `2 * he`. Für allgemeine Shapes (PointNet-Encoder später) wird der gleiche Skalar aus dem Surface-Sample / decoder berechnet.

Konkret hinzufügen:
- `extent_xyz_0` und `extent_xyz_1` als Node-Features (3 Skalare pro Knoten)
- node_dim wächst von 11 auf 14

Damit hat das GNN explizit Zugang zur Geometrie-Information, die es aktuell aus dem 8-dim Latent rekonstruieren muss. Die Edge-Features bleiben unverändert (`Δx, Δy, Δz, dist_xy`), so dass die Edge-Schicht **shape-agnostisch** bleibt.

**Warum das nicht das Constraint verletzt**:
- `extent` ist für jede Shape definierbar (Boxen, Zylinder, beliebige convexe oder nicht-convexe Geometrien). Für PointNet-Encoded Shapes berechenbar aus den Surface-Samples. Für SDF-Decoded Shapes aus dem Bounding-Box-Range.
- Im Repair-Loop: solange Block1's Latent `z_1` verändert wird, bleibt sein `extent` an die Geometrie gekoppelt — wir müssen das `extent` aus `z_1` ableiten (Decoder oder Encoder-inverse oder pre-computed lookup).

**Repair-Konsistenz** (kritisch): Im aktuellen `repair_optimizer.py` werden Block1-Latent UND Block1-Position optimiert. Der Block1-Extent muss bei Latent-Änderung mitlaufen. Zwei Optionen:
1. **Differenzierbar via Decoder**: `extent_z_1 = decoder_extent(z_1)`. Erfordert einen kleinen Decoder-Head oder eine Mini-Sampling-Routine. Differenzierbar, aber teuer.
2. **Detached approximation**: `extent` wird als detached Constant aus dem **initialen** `z_1` berechnet und während Repair fixiert. Verlust an Genauigkeit (Latent-Drift verändert echte Geometrie, Feature läuft nicht mit), aber einfach.
3. **Über BoxEncoder-Inversion**: BoxEncoder ist `half_extents → z`. Die Inversion (Latent → half_extents) wäre ein neues Module. Funktioniert nur für Boxen → verletzt Constraint.

Option 1 ist die saubere shape-agnostische Lösung. Im Trainings-Schritt brauchen wir das nicht (extents sind im Datensatz vorberechnet); im Repair-Schritt schon.

**Code-Änderungen**:
- **Datensatz-Vorbereitung**: `src/gnn_dataset_preparation.py` erweitern, Block-Surface-Bounding-Box berechnen (für Boxen trivial: `extent = block_size`; vorbereiten für PointNet-Erweiterung).
- **Node-Feature**: in `build_graph` von 11 auf 14 erweitern.
- **Config**: `config/config.yaml` `node_dim: 11 → 14`.
- **Spiegel-Stellen**: `analyze_label_generalization.py:build_batch_graph_inputs`, `repair_optimizer.py:build_*`, `repair_process.py:*`.
- **Repair**: Decoder-basierte differenzierbare Extent-Berechnung implementieren, ODER detached approximation als Phase-1-Compromise (mit klarem Kommentar, dass es Tech-Debt ist).
- **Datengenerierung**: `src/dataset_generation.py` ergänzen, dass `extent_xyz` pro Block in den Datensatz geschrieben wird (für Boxen trivial; für spätere allgemeine Shapes erweitert sich an genau dieser Stelle).

**Erwarteter Effekt**: Drastische Reduktion der Gap-MAE im kritischen Bucket (Schätzung <0.15 mm), weil das GNN die Geometrie-Information explizit hat statt sie aus dem 8-dim Latent rekonstruieren zu müssen.

**Risiken / Caveats**:
- **Repair-Konsistenz** (oben): Phase-1 mit detached extent vereinfacht das Coding, hat aber bekannte Limitierung. Phase-2 wäre differenzierbarer Decoder-Pfad.
- Wenn der Effekt klein ist, war die Latent-Decodierung doch nicht der Flaschenhals — dann zu H2 weiter.

**Aufwand**: 3-4 h Code (mehr als ursprünglich, weil Repair-Konsistenz mitgedacht), 1 Trainingslauf, 1 Repair-Validation.

### H2 — Threshold-relative signed Parametrisierung mit Boundary-Hinge-Loss

**Idee**: Mag/Sign-Split aufgeben, signed `gap` direkt regressieren, mit Loss, der das Verhalten am Threshold explizit penalisiert:
- Hauptloss: Huber auf signed `gap` in std-space.
- Hinge-Term: `L_thresh = max(0, threshold − pred * sign(true)) ** 2` für true-feasible Samples symmetrisch.
- Konzeptuell ähnlich Fabios früherem Versuch, aber mit explizit korrekter Loss-Konstruktion.

Loss-Signal ist shape-agnostisch — es greift nur auf den signed gap-Wert zu.

**Code-Änderungen**:
- `src/gnn.py:139-149` — `mlp_gap_mag` + `mlp_gap_sign` durch ein `mlp_gap` ersetzen (signed std-output).
- `src/gnn_training.py:135-214` — `compute_reg_loss` umschreiben.
- `src/repair_optimizer.py:428-435` — Inferenz-Pfad vereinfachen.
- `src/label_analysis/analyze_label_generalization.py:362-388` — analog vereinfachen.

**Erwarteter Effekt**: Mittlere Verbesserung der Gap-MAE (Schätzung 0.25-0.30 mm), bessere Threshold-Agreement (97-98 %).

**Risiken**:
- Verliert die boundary-mass-Information, die der Mag/Sign-Split einbringen sollte.
- Hinge-Term-Skala muss kalibriert werden.

**Aufwand**: 2-3 h Code, 1 Trainingslauf.

### H3 — Auxiliärer "Gap-Feasibility"-Klassifikations-Head

**Idee**: Bestehenden mag/sign-Pfad behalten, zusätzlich `mlp_gap_feasible` Binary-Head ergänzen, der direkt `|true_gap| < threshold` klassifiziert. Im Repair als zusätzliches `p_gap`-Signal.

Shape-agnostisch — der Head arbeitet auf dem Trunk-Output, nicht auf box-spezifischen Features.

**Erwarteter Effekt**: Klein-bis-mittel. Adressiert das Symptom (Bluff im Repair), nicht die Ursache (schlechte Magnitude).

**Aufwand**: 1-2 h Code, 1 Trainingslauf.

## Empfehlung

**H1 (revidiert) als erstes Experiment.** Begründung:
1. Größter erwarteter Hebel auf Gap-MAE.
2. Adressiert die strukturelle Lücke (Latent ist mit 8 Dim zu klein, um Geometrie verlustfrei zu kodieren — explizite Extent-Feature short-circuited das).
3. Erweitert den Datensatz und die Pipeline an genau der Stelle, wo später PointNet-Shapes andocken werden — also Forward-kompatibel.
4. Branch-isoliert.

H2 als zweites, falls H1 unzureichend. H3 nur als Ergänzung, nicht als Standalone.

## Workflow für H1

### Vor-Phase — Daten + Slurm

1. **Neuen Datensatz generieren**: `src/dataset_generation.py` modifizieren, dass `extent_xyz` pro Block (= `Block_SizeXYZ` für Boxen, vorbereiteter Hook für PointNet-Shapes) als zusätzliche Spalten geschrieben wird. Generieren auf Euler oder lokal (laufzeitabhängig). Neuer Filename mit aktuellem Datum.
2. **GNN-Slurm-Script**: Aktuell nur auf Euler vorhanden. Fabio committed das Slurm-Skript auf den neuen Branch (oder schickt mir den Inhalt; ich schreibe dann eine Version analog zu `autoencoder_train_twostage.slurm`).

### Code-Phase

1. **Branch anlegen**: `git checkout -b feature/gnn-shape-extent-features` (von `feature/gnn-optimization`).
2. **Code-Änderungen** (Reihenfolge):
   1. `src/dataset_generation.py` — `extent_xyz` pro Block schreiben.
   2. Neuen Datensatz generieren (kleines Subsample lokal als Smoke-Test, vollständiger Lauf später).
   3. `src/gnn_dataset_preparation.py:217-222` — Node-Feature von 11 auf 14 erweitern.
   4. `config/config.yaml:65` — `node_dim: 14`.
   5. `src/label_analysis/analyze_label_generalization.py:152-160` — `build_batch_graph_inputs` analog erweitern.
   6. `src/repair_optimizer.py` — Build-Stellen + extent-im-Repair: für Phase 1 detached extent (mit `# TODO: differentiable decoder-based extent for general shapes` als Kommentar).
   7. `src/repair_process.py:430+` — analog.
   8. `tools/smoke_test_graph.py` lokal grün.
3. **Sanity-Check**: Override-Flag in `src/gnn_training.py` (z.B. CLI-Arg `--quick` oder env-Variable `GNN_QUICK=1` → 5k Samples, 5 Epochen). Lokal 5-10 Min CPU laufen lassen, Loss muss runter gehen.
4. **Commit + Push** (nach Bestätigung).

### Cluster-Phase

5. **Auf Euler**: Fabio macht `git pull` + `sbatch <gnn-training-script>`. Wandb offline, Sync später.
6. **Auswertung**: nach Sync → `analyze_label_generalization.py` mit neuem Checkpoint. Vergleichen mit Erfolgskriterien unten.
7. **Repair-Validation**: Bluff-Rate-Skript von `feature/new-architecture` cherry-picken (auf neuen Branch) oder auf der alten Branch ausführen → Vorher/Nachher vergleichen.

### Erfolgskriterien

| Metrik | aktuell | Ziel | Mindest-Erfolg |
|---|---|---|---|
| planned_gap MAE Bucket `[0,3)` | 0.297 mm | < 0.15 mm | < 0.20 mm |
| planned_gap Threshold-Agreement | 94.24 % | > 97 % | > 95.5 % |
| under_surface / overlap / thickness MAE | unverändert | unverändert | nicht schlechter als 110 % aktuell |
| Repair-Bluff-Rate | TBD messen | TBD halbieren | TBD reduzieren |

### Abbruchkriterien

- H1 verschlechtert andere Labels signifikant (>10 % MAE-Anstieg) → Edge-Feature nimmt Trunk-Information weg → H2 versuchen.
- H1 verbessert Trainingsmetriken, aber Repair wird schlechter → Latent-Pfad war für Gap-Repair gebraucht → entweder differenzierbarer Decoder-Extent (Phase 2 von H1) oder zurück zu H2.

## Antworten auf die offenen Fragen

1. **Sanity-Check**: Override in `gnn_training.py` (env-Variable oder CLI-Arg), nicht neues Skript. Weniger Code-Sprawl, lebt im Trainings-Skript selbst.
2. **GNN-Slurm-Script**: existiert auf Euler, wird beim Branch-Setup mit committed. Falls nicht praktikabel, schreibe ich eine Variante analog zu `autoencoder_train_twostage.slurm`.
3. **Bluff-Rate-Skript**: existiert auf `feature/new-architecture`. Ich werde es vor der Auswertung auf den neuen Branch cherry-picken (oder lokal aus dem anderen Branch ausführen, je nach Pfadabhängigkeiten).
4. **Datensatz**: neuen generieren (Step 1 oben) und ihn benutzen. Filename mit Datum 2026-05-08.

## Inhaltlicher Punkt zum Diskutieren

Das Repair-Konsistenz-Problem (Extent-Feature läuft im Repair nicht mit, wenn `z_1` sich ändert) ist real. Phase-1-Lösung "detached extent" ist Tech-Debt. Phase-2 wäre ein **kleiner differenzierbarer Decoder-Head**, der `z → extent_xyz` mappt — für Boxen trivial (Encoder-Inversion mit BoxEncoder), für PointNet-Shapes ein eigenes Mini-Modul.

Wenn Fabio vor dem Code-Start sagt "ich akzeptiere Phase-1 als Tech-Debt für jetzt, aber dokumentiere es", machen wir Phase 1. Wenn er sagt "macht keinen Sinn ohne den differenzierbaren Pfad", machen wir Phase 1+2 zusammen → mehr Aufwand (~1-2 h zusätzlich), aber sauberer.
