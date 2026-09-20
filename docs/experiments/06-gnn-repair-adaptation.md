# Experiment 06 — GNN + Repair Adaptation for the Sim Dataset (Plan)

**Code branch:** `feature/ae-shape-vocab-14` · **Stand: 2026-07-24 — Schritte 0–5 UMGESETZT (Repair Weg 1, 80 %); Schritt 6 (z-Kontinuum) begonnen: Generator gebaut+gepusht, Sim-Lauf offen.**
Baut auf `05-sim-dataset-generation.md` (Sim-Datensatz) auf. Referenz-CSV: der letzte Sim-Lauf
(gleiche Struktur wie der große Lauf).

---

## Ausgangslage (gute Nachricht)
Die bestehende Shape-GNN-Pipeline ist schon fast passend:
- `src/shape_gnn_dataset.py`: Knoten = **`[z(latent) | bbox(3) | node_type(2)]`** — genau die Zielstruktur. `z` = PointNet-Encoding der **kanonischen** Shape, `bbox` = metrische Größe. `pair_to_graph()` baut einen 2-Knoten-Graph, `node_dim()` = latent+3+2.
- `src/gnn.py`: Late-Fork-GATv2, Heads = `reg_out[overlap_std, thickness_std]`, `binary_out[Assembly_Good]`, `mode_logits[overlap_fail, thickness_fail]`.
- `src/gnn_training.py`, `src/shape_repair_optimizer.py` (scale-Repair) existieren.

**Was NICHT passt:** (1) Datenquelle = alter synthetischer Generator statt Sim-CSV; (2) Targets = nur overlap/thickness (Regression) statt der 4 Sim-Flags; (3) die Sim-CSV hat **keine Größen**.

---

## ⚠️ PREREQUISITE (Sim-Seite, VOR dem großen Lauf) — Schritt 0
Die Sim-CSV enthält nur `Obj{i}_Pos{XYZ}` + `Obj{i}_UsdName` + Flags — **keine `Obj{i}_Size{XYZ}`**. Der GNN-Knoten braucht die `bbox` (Größe). → **`save_data.py` um Size-Spalten erweitern** (aus `self.object_sizes`). Andernfalls ist der große Datensatz für den GNN unbrauchbar.
- verify: neue CSV hat `Obj0_SizeX/Y/Z`, `Obj1_SizeX/Y/Z`; Werte ~0.005–0.05 m, variieren.
- **Status: ERLEDIGT (2026-07-23, Commit `07b2b00`).** `save_data.py` schreibt jetzt `Obj{i}_Size{XYZ}` aus `self.object_sizes`. Verifikation steht mit dem nächsten Sim-Lauf aus (Spalten in der CSV prüfen).

---

## Schritt-Plan (jeder Schritt: umsetzen → verifizieren → hier als Status festhalten)

### Schritt 1 — 14 kanonische `z` vorberechnen
Da der Encoder größen-invariant ist, ist `z` **pro Shape-Typ konstant**. Einmal die 14 kanonischen Shapes (SDF → Oberflächenwolke → `to_canonical` → Encoder) mit dem LV-Checkpoint encodieren → **Lookup `{shape_name: z}`** (14 Vektoren).
- Codec: `Code/encoder_decoder_model/best_encoder_decoder_general_canon_lv_v14_lam005_latentdim32_active9.pth`.
- verify: 14 distinkte z (paarweise cos < 1), z-Norm plausibel; Round-Trip decode → |SDF|@surface sub-mm für 12/14.
- **Status: ERLEDIGT (2026-07-23).** `tools/precompute_shape_latents.py` → `encoder_decoder_model/shape_latents_lv_v14_lam005.pt` ({usd_name: z}). 14 distinkt (max cos 0.998 = box~rounded_box, erwartet), z-Norm 0.20–0.34.
- **OFFEN:** die 2 Beispiel-Prismen (Triangular/Hexagonal) fehlen im Lookup (kein SDF) → entweder aus `obj_files/` entfernen (empfohlen, sauberes 14-Vokabular) oder Meshes separat encodieren.

### Schritt 2 — CSV → Graph-Loader
Neuer Loader `src/sim_gnn_dataset.py` (analog `shape_gnn_dataset.pair_to_graph`): pro CSV-Zeile 2-Knoten-Graph.
- Knoten `[z | bbox | type]`: `z` = Lookup(Obj{i}_UsdName), `bbox` = `Obj{i}_Size` (aus Schritt 0), `type` = one-hot (Basis/Kind).
- Kanten: gerichtete 2-Knoten-Edges mit Δpose aus `Obj{i}_Pos` (wie Box-Pipeline `_edge_features`).
- Labels (Multi-Label, binär) vom **aktiven Knoten** (ObjNew): `[OBJECT_GAP, OVERLAP_INSUFFICIENT, THICKNESS_EXCEEDED, TIPPING]` + `Assembly_Good?`.
- verify: baut `Data`-Objekte, `node_dim == latent+3+2`, Label-Verteilung == CSV-Analyse (z.B. feasible ~25%).
- **Status: ERLEDIGT (2026-07-23).** `src/sim_gnn_dataset.py`: `csv_to_graphs(csv, latents)` → Liste `Data` (x(2,37)=z32+bbox3+type2, edge_attr(2,6), y(1,4)=[gap,overlap,thickness,tipping], y_good(1,1)). Prism-Zeilen (eines der 2 Objekte Prisma) werden übersprungen → ~76% bleiben (~510k von 672k). Getestet an 1613: Assembly_Good ~30%, alle 4 Targets mit Signal.

### Schritt 3 — GNN-Heads anpassen
Statt Regression overlap/thickness → **4 binäre Failure-Heads** (gap, overlap, thickness, tipping) + `Assembly_Good`-Head. Late-Fork-Trunk bleibt. Flat-Gating: overlap/thickness-Head-Loss für runde Shapes maskieren (Label = „n/a"), oder Loss nur auf flachen Knoten.
- verify: Forward-Pass-Shapes stimmen; kein Head-Collapse.
- **Status: ERLEDIGT (2026-07-23).** Neue Klasse `SimAssemblyGNN` in `src/gnn.py` (die bestehende `GNN` bleibt unberührt — von ~15 Dateien importiert). Late-Fork-GATv2-Trunk + active-node-Readout (`x[1::2]`), Heads: 4-Logit-Failure `[gap,overlap,thickness,tipping]` + 1-Logit Assembly_Good. `forward(x,edge_index,edge_attr) -> (fail(B,4), good(B,1))`. Verifiziert.

### Schritt 4 — Training
`src/gnn_training.py` adaptieren: Multi-Label-BCE über die 4 Targets + Assembly_Good, Klassen-Gewichte gegen Imbalance, 80/10/10-Split, Euler-Slurm.
- verify: val per-Target-F1 vernünftig (v.a. tipping/overlap/thickness), keine Kollapse; feasible-Klasse lernbar.
- **Status: FERTIG (2026-07-24), auf Euler trainiert.** `src/sim_gnn_training.py` (BCEWithLogits + pos_weight je Target aus Train-Split, 80/10/10, Adam, Early-Stop auf val-Loss, Checkpoint `gnn_models/sim_gnn_shape_node37_32.pth`) + `sim_gnn_train.slurm`.
- **Zwei Verbesserungen gegenüber dem Plan:** (A) **Input-Standardisierung** (train-split Mean/Std als `register_buffer` im Modell → landet im state_dict, Repair erbt es). War der große Hebel: mm-Signale (bbox/pos ~0.005–0.03 m) gingen neben z-Latents O(1) unter. (B) Metrik = **AUROC/AP + val-getunte Thresholds** statt fix 0.5 (Thresholds im ckpt).
- **OBJECT_GAP als Target GESTRICHEN** (Head jetzt 3: overlap/thickness/tipping). Belegt nicht lernbar: analytischer Gap `dz−s0z/2−s1z/2` AUROC 0.51 auf allen 508k, GNN 0.53. Der Sim berechnet das Flag aus nicht-exportiertem Zustand.
- **Finale Test-Ergebnisse (Early-Stop E39, 3 Targets):** OVERLAP F1 0.831/AUROC 0.953/AP 0.925 · THICKNESS 0.992/1.000/1.000 (trivialer bbox-Read) · TIPPING 0.627/0.788/0.616 · **Assembly_Good 0.684/0.869/0.713**. Per-Shape-Auswertung: `tools/sim_gnn_eval_by_shape.py`.

### Schritt 5 — Repair gegen den neuen GNN
**Status: UMGESETZT (2026-07-24), Weg 1.** Eigener `src/sim_repair_optimizer.py` (nicht `shape_repair_optimizer`, da SimAssemblyGNN andere Heads/Standardisierung hat).

**`repair_pair`:** Adam auf Kind-Variablen durch eingefrorenen GNN. Top-Down-Stapel → `pos_z` per differenzierbarem Kontakt-Snap fix, nur `pos_xy` frei; Größe = **uniformer Skalar `s`** (bbox=bbox0·s, on-manifold, weil die Sim uniform 0.15–1.0 skalierte); `z` optional. Loss = good_logit hoch + 3 Failure-Logits runter + Drift-Regularisierung. Standardisierung aus den ckpt-Buffern. Analytischer overlap/thickness-Check (AABB) = partielle Ground Truth.

**Kern-Entscheidung (Weg 1):** Die Form ist beim Sim-GNN **kategorial** (14 diskrete z, größen-invariant) → **kein z-Gradient**. Stattdessen `repair_with_shape_search`: Original-Form zuerst (scale+pos), sonst die flachen Shapes diskret durchprobieren, feasible mit kleinster Design-Abweichung nehmen. (Warum kein z: siehe „z-Repair" unten.)

**Ergebnisse** (echter ckpt, `tools/sim_repair_demo.py`, N=40 infeasible flat-active):

| Modus | feasible | Bluff | p_good vor→nach | z_drift |
|---|---|---|---|---|
| `scale` (size+pos, z fix) | **37.5 %** | 5 % | 0.105→0.327 | 0 |
| `scale+z` (off-manifold) | 17.5 % | **80 %** | 0.105→0.959 | 0.10 |
| `shape-search` (Weg 1) | **80 %** | — | — | 67.5 % Formwechsel |

- `scale` = ehrlich, funktioniert (Bluff 5 %). `scale+z` = **Bluff am echten Modell bestätigt** — GNN-Konfidenz →0.96, aber reale Feasibility SINKT auf 17.5 %, weil der Optimizer off-manifold-z ausnutzt (z_drift nur 0.10 reicht, GNN scharf auf die 14 z). `shape-search` = Gewinner (37.5→80 %).
- **z-Repair-Diagnose:** LV/Encoder löst das NICHT (2 Manifolds — LV betrifft den Encoder-Manifold, das Problem ist der diskrete z-Support des GNN). Box-Latent-Repair funktionierte nur wegen z↔size-Bijektion, nicht wegen LV. Verteidigbares Negativergebnis.
- **VORBEHALT / OFFEN:** `feasible` = analytisch overlap+thickness UND GNN-good-**Surrogat**; **tipping + Assembly_Good sind NICHT Isaac-verifiziert** → 80 % ist eine surrogat-gestützte Obergrenze, Tipping-Bluffs offline unsichtbar. **Nächster echter Schritt: Isaac-Re-Check** der shape-search-Configs auf Euler = einzige belastbare Endvalidierung.

---

## Schritt 6 — z-Kontinuum (z-Repair ehrlich machen)
**Status: ABGESCHLOSSEN (2026-07-25). Ergebnis = z-Repair blufft auch mit
Continuum-GNN + LV-Maske, real bestätigt in Isaac (5 % feasible vs. 35 % ehrlich).
Siehe „Ergebnisse (Schritt 6)" unten.**

**Problem:** Der Sim-GNN sah `z` nur als **14 diskrete Lookup-Vektoren** (ein z pro Shape-Typ, größen-invariant). Ein z-Gradient läuft sofort aus diesem Support → `scale+z` blufft (p_good→0.96, reale Feasibility sinkt, 80 % Bluff). **LV löst das nicht** (zwei Mannigfaltigkeiten: LV betrifft den *Encoder*-Manifold; das bindende Problem ist der *diskrete z-Support des GNN*). Box-Latent-Repair ging nur wegen z↔size-Bijektion.

**Lösung:** Den GNN auf einem **Form-Kontinuum** trainieren, damit `z` eine echte kontinuierliche Eingabe wird. Trainingsformen per **Decoder aus gejittertem z** erzeugen (dann ist jedes z bekannt, kein Nach-Encodieren):
```
EINMALIG:   14 Formen → Encoder → 14 Basis-z            (schon vorhanden)
DATEN-GEN:  Basis-z jittern → DECODER → Mesh → Isaac → Labels   (z bekannt)
TRAINING:   (z, bbox, pos) → GNN → Feasibility          (jetzt dichtes z)
REPAIR:     Form → Encoder → z → GNN → z optimieren → Decoder → Form
```
Nur der **Decoder** läuft auf der Sim-/GPU-Seite; der Encoder bleibt im Code-Repo (Basis-z + Repair-Laufzeit).

**Wie das GNN die Kontinuum-z bekommt:** NICHT aus dem Dataset. Die CSV liefert nur den **Namen** (`Obj{i}_UsdName` = `box__j0007.usd`); das z steht in der **`{Name→z}`-Tabelle** (`latents_continuum.pth`, vom Generator, z gratis bekannt). Der Loader macht den **Join** `Name → Tabelle → z` — genau der bestehende `latents[UsdName]`-Lookup, nur mit größerer Tabelle. **Schritt 3 = die Basis-14-Tabelle + `latents_continuum.pth` vereinigen** (beide `{Name→z}`-Dicts), damit auch die Basis-Meshes (`box.usd`) im Pool ein z haben.

**Teilschritte:**
1. **Mesh-Generator** ✅ (Sim-Repo `scripts/z_continuum/`, gepusht `origin/dataset_stls-z-continuum` @`0c8eedc`): eigenständig (nur torch/numpy/skimage/trimesh + 4.4 MB LV-Decoder-`.pth`, kein LFS), jittert die 14 Basis-z → Decoder → Marching Cubes (res 64) → `<base>__jNNNN.obj` + `latents_continuum.pth`. Lokal getestet: wasserdicht, korrekte cm-Größen, Proportionen (→ z) variieren. Generierte OBJs + z-Tabelle **gitignored** (nie auf GitHub). Sim-Fix `scene_utils.py`: `is_flat` per Basis-Form vor `__` (verifiziert einzige namensabhängige Stelle).
2. **Continuum-Sim** ✅ (GPU, num_envs 1000): 777 Continuum-Meshes (13 Shapes; **Cone fällt raus** — Spitze jittert immer non-watertight). CSV 608k → `.txt` via `sim_csv_to_txt --latents latents_continuum.pth` (Legacy-`.usd` ohne `__j` automatisch gefiltert) = **580 640 saubere Zeilen** (`sim_shape_continuum_2027.txt`), feasible 31.7 %.
3. **GNN-Loader** ✅ (`feature/z-continuum-meshes` @`0e683b8`): `load_shape_latents` nimmt eine Tabellen-Liste, vereinigt (Basis-14 + Continuum) → **791 Meshes**, z pro Mesh, node_dim 37.
4. **GNN retrainen** ✅ (Euler): Test OVERLAP AUROC **0.957**, THICKNESS 1.0, TIPPING **0.815**, Assembly_Good **0.895** — **alle ≥ Basis-Lauf** (14-z: 0.953/0.788/0.869). Kontinuum kostet keine Qualität + gibt dichten z-Support.
5. **z-Repair-Ablation** ✅ — siehe „Ergebnisse (Schritt 6)".

**Branches:** Sim = `dataset_stls-z-continuum` (IDEALLab, gepusht). Code = `feature/z-continuum-meshes` (von `feature/ae-shape-vocab-14`).

---

## Ergebnisse (Schritt 6) — z-Repair blufft, real bestätigt

**Zwei Mannigfaltigkeiten, getrennt getestet:** *Manifold 1* = Encoder (welche z → gültige Formen; LVs Domäne). *Manifold 2* = z-Support des GNN (welche z hat er gesehen; Domäne des Kontinuums). Der Bluff sitzt in **Manifold 2**.

**Offline-Ablation** (`tools/sim_repair_demo`, Continuum-GNN, n=100; feasible = analyt. overlap+thickness ∧ GNN-good):

| mode | p_good→ | feasible | bluff | z_drift |
|---|---|---|---|---|
| scale (z fix) | 0.49 | **35 %** | 27 % | 0.000 |
| scale+z (32d) | 0.97 | 19 % | **80 %** | 0.082 |
| scale+z (14 LV-aktive Dims) | 0.95 | 16 % | **84 %** | 0.152 |

→ **LV-Aktive-Dim-Maske hilft NICHT** (Bluff 84 % ≥ 80 %; Exploit lebt in den aktiven Dims = Manifold 2). **Continuum-GNN macht z-Repair NICHT ehrlich** (~80 % wie beim 14-z-GNN).

**Isaac-Re-Check** (echte Physik; Export → decode z → deterministischer Replay → `compare`; n=100/Modus, dedupe per EnvIndex):

| mode | real_feasible (Isaac) | FAILED | OVERLAP-fail | TIP |
|---|---|---|---|---|
| scale (ehrlich) | **35 %** | 65 % | 32 % | 43 % |
| scale_z (z-repair) | **5 %** | 95 % | 68 % | 52 % |

→ **scale: real 35 % == offline 35 %** (ehrlicher Repair hält; Offline-Schätzung vertrauenswürdig). **scale_z: GNN gab p_good ≈ 0.96, real nur 5 % feasible** → Bluff unter echter Physik bestätigt, sogar schärfer als offline. OVERLAP-fail **verdoppelt sich** (32 → 68 %): die z-Form-Morphs erzeugen keine echte Auflagefläche — GNN getäuscht, Physik nicht.

**Fazit:** z-/Latent-Form-Repair ist ein Bluff, **4-fach abgesichert** (14-z-GNN → LV-Maske → Continuum-GNN → Isaac). **Ehrliches Repair = scale+pos (bbox), Form via diskreter Suche** — real validiert (35 %). **LVs Wert ist damit sauber abgegrenzt:** Kompression (eff. ~10 aktive Dims) + sauberer Continuum-Enabler + „Compressing Latent Space" selbst — **nicht** z-Repair-Retter.

**Pipeline-Dateien:** Export `tools/sim_repair_export.py`; Decode `Sim scripts/z_continuum/decode_repairs.py`; Replay `Sim …/pose_orientation_two_robots/replay.py` (+ Patches `move_object.py`, `randomize_objects.py`, alles hinter `REPLAY_CONFIG`-Guard); Vergleich `Sim scripts/z_continuum/compare_replay.py`. LV-Maske: `sim_repair_optimizer.active_latent_dims` + `repair_pair(active_dims=)`.

---

## Referenz-Dateien
- Datensatz: `05-sim-dataset-generation.md`, Sim-Branch `dataset_stls`.
- GNN: `src/shape_gnn_dataset.py`, `src/gnn.py`, `src/gnn_training.py`.
- Repair: `src/shape_repair_optimizer.py`, Stats `tools/shape_repair_stats.py`.
- Encoder: `src/validate_shape_encoder.py` (`to_canonical`, encode), LV-Checkpoint s.o.

## Reihenfolge
Schritt 0 (Size-Spalten) → großer Datensatz-Lauf → Schritte 1–5. Schritt 1 (z-Vorberechnung) und der Loader (2) lassen sich **vorbereiten**, während der Lauf läuft.
