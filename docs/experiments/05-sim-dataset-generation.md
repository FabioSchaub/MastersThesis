# Experiment 05 — Part III Sim Dataset Generation (Isaac Sim)

**Code branch:** `feature/ae-shape-vocab-14` · **Sim branch:** `dataset_stls` (IDEALLab/Woodworking_Simulation)
**Stand: 2026-07-23.** Ergänzt `04-vocab-and-targets.md` (Codec) und `dataset-stls-sim` (Sim-Bring-up).

---

## TL;DR
Teil-3-Datensatz wird in **Isaac Sim** erzeugt (Branch `dataset_stls`), auf dem GPU-Rechner. Modell = **2-Block-Top-Stacking**: Basis auf dem Tisch, zweites Teil oben drauf, von oben verschraubt. **14 Shapes** (als OBJ, LFS), **uniforme Größen-Randomisierung** pro Objekt. **4 Feasibility-Targets** mit Signal: `OBJECT_GAP`, `OVERLAP_INSUFFICIENT`, `THICKNESS_EXCEEDED`, `TIPPING` (+ `Assembly_Good?` = UND). Physik/Verteilung verifiziert (Lauf „1517", 69k Zeilen). **Datengenerierung steht.** Nächster Schritt = GNN aufs neue Dataset umstellen (siehe unten).

---

## Ziel & Modell
- **2 Blöcke, Top-Stacking** (Fabio-Entscheidung): das Kind (`obj_new`) wird auf die **+Z-Fläche** der Basis gestapelt. Grund: geschraubt wird **von oben** → nur so sind alle Targets kohärent.
- Der **horizontale (X-Y) Offset** des Kinds steuert gemeinsam `overlap` und `tipping`.
- Kollision (`OBJECT_OBJECT_CONTACT`) braucht n≥3 → **raus**. Tool-Interferenz (Greifer/Schraubendreher) braucht `cfg.standalone*`-Modi → **aus** → Flags = 0.

## Die 4 Targets (Schwellen)
| Flag | Definition | Schwelle | nur flach? |
|---|---|---|---|
| `OBJECT_GAP` | Kind berührt Basis nicht (vertikaler Abstand) | > 3 mm | nein |
| `OVERLAP_INSUFFICIENT` | zu kleine horizontale (X-Y) Auflagefläche für Schraube | < 10 mm (je Achse) | **ja** |
| `THICKNESS_EXCEEDED` | Kind zu hoch für Top-Schraube (Z-Höhe) | > 20 mm | **ja** |
| `TIPPING` | Stapel kippt (Eigen-Orientierung) | `dot(up, z) < 0.995` | nein |
| `Assembly_Good?` | UND aller aktiven Checks | — | — |

**Flat-Gating:** runde Shapes (`sphere, ellipsoid, cylinder, half_cylinder, capsule, cone`) haben keine flache Schraubfläche → overlap/thickness werden für sie **nie** gesetzt (`ROUND_SHAPES` in `scene_utils.py`, gegated in `failure_utils.py`). Prismatische/flache Shapes (box, rounded_box, hex_prism, wedge, i/u/l/t_profile) werden gewertet.

## Shapes → OBJ
- 14 Shapes werden per **`Code/tools/export_shape_objs.py`** aus den analytischen SDFs (`src/shape_primitives.py` + box/sphere/ellipsoid aus `src/enc_dec_dataset_generation.py`) per Marching Cubes exportiert.
- **In cm exportiert** (`--scale 100`), weil der Isaac-OBJ→USD-Konverter OBJ-Zahlen als **cm** interpretiert (×0.01 → m). Liegen in `obj_files/` des Sim-Branches (Git LFS).

## Größen-Randomisierung
- Pro Objekt **uniformer** Scale-Faktor `_OBJ_SCALE_MIN..MAX = 0.15..0.5` (`move_object.py`), form-erhaltend → variiert die bbox (nicht `z`, das = Form).
- Angewandt als **expliziter USD-Scale-xformOp**, der mit der Einheiten-Skala des Konverters **multipliziert** wird (nicht überschreibt — sonst 100× zu groß → Physik-Explosion).
- Die gemessene USD-Bbox (`compute_object_sizes_from_usd`) trägt die Größe in alle Checks.

## CSV
- Ort: **`<cwd>/data/num_objects_2_<YYYYMMDD_HHMM>.csv`** (relativ zum Startordner).
- Schema (pro Zeile = eine Assembly): `DataPoint, EnvIndex, Obj{i}_UsdName, Obj{i}_Pos{XYZ}, ObjNew_Screwdriver{XYZ}, ObjNew_TableInterferenceVal, Obj{i}_ScrewdriverImpact*, Obj{i}_<FAILURE_NAME> (je Objekt & Flag), Assembly_Good?`.

## Verifizierte Verteilung (Lauf 1517, 69k Zeilen)
feasible **25.5%** / infeasible 74.5%. Failure-Raten (ObjNew): OVERLAP 37.5% (flat 59.6%), TIPPING 32.3%, THICKNESS 18.8% (flat 29.9%), GAP 11.9%, TABLE/CONTACT 0%. Gating korrekt (round: overlap/thickness 0%). #Fehler/Zeile: 0=26% · 1=50% · 2=23% · 3+=1%.

## Großen Datensatz erzeugen
Auf dem GPU-Rechner, Branch `dataset_stls`:
```
git pull
<IsaacLab>/isaaclab.bat -p scripts/random_agent.py --task=Template-Pose-Orientation-Two-Robots-Direct-v0 --num_envs 1000 --headless
```
Laufen lassen bis die CSV **~300–500k** Zeilen hat (paper ~174k als Minimum), dann stoppen. CSV liegt in `<cwd>/data/`.

## Bugs, die beim Bring-up gefixt wurden (chronologisch, alle auf `dataset_stls`)
1. `NameError num_objects` in `config_class` (Class-Body-Comprehension) → modul-globale `NUM_OBJECTS`.
2. `omni.kit.asset_converter` nicht aktiviert → per Extension-Manager on-demand aktivieren.
3. **OBJ→USD-Unit:** Meshes in Metern → 100× zu klein → in **cm** re-exportiert.
4. `TABLE_INTERFERENCE` dominierte (42%) → Kind-Platzierung auf/über Tisch geclampt.
5. `--num_envs > 1024` → IndexError → `object_info` zur Laufzeit auf `num_envs` neu bauen.
6. `THICKNESS` = **Z-Höhe** (von oben schrauben), nicht Kontaktflächen-Achse.
7. **Größen-Scale:** expliziter xform-Scale, der die Konverter-Einheiten-Skala **multipliziert** (nicht überschreibt — sonst 4-m-Objekte + Physik-Explosion). ← der große Fix.
8. Top-Stacking (+Z) statt zufälliger Flächen → alle 4 Targets kohärent.

## Bekannte Caveats / TODO(fabio)
- **Debug-Prints** (`[dbg]`, `[shape-debug]`) sind noch im Code (nur Konsolen-Spam, verfälschen die CSV nicht). Vor Finalisierung raus.
- **Joint-Snap-Warnung** bleibt, ist aber harmlos (~7 mm Setzung; planned≈actual verifiziert).
- **feasible 25%** — via `offset_factors`-Range (aktuell ±0.8) tunebar (kleiner → mehr feasible).
- **Scale-Range 0.15–0.5** tunebar; beeinflusst overlap/thickness-Balance.
- overlap/thickness nur für flache Shapes definiert (bewusst gegated).

## Gewählter LV-Codec (für die GNN-Knoten)
`Code/encoder_decoder_model/best_encoder_decoder_general_canon_lv_v14_lam005_latentdim32_active9.pth` (9-aktiv, 12/14 sub-mm, `latent_dim 32`, eff. ~9–10 Dims). Der Trainingslauf erzeugte zwei Checkpoints (`20260630-131058` und `-131141`); der hier abgelegte ist der erste, der zweite ist **nicht** der der Thesis. Der Zeitstempel wurde beim Ablegen im Repo durch `active9` ersetzt.

---

## NÄCHSTER SCHRITT — GNN aufs neue Dataset anpassen
Der Box-GNN passt nicht direkt. Nötig:
1. **Neuer Dataset-Loader:** Sim-CSV → Graphen. Pro Objekt: Shape-Typ + Größe + Pose + Labels aus der CSV.
2. **Knoten `[z | bbox | type]`:** `z` = LV-Codec-Encoding der **kanonischen** Shape. Da der Encoder größen-invariant ist, ist `z` **pro Shape-Typ konstant** → einmal die 14 kanonischen Shapes encodieren (14 z-Vektoren, Lookup). `bbox` = gemessene Größe, `type` = Shape-Typ/Node-Rolle.
3. **Kanten:** 2-Knoten-Graph (Basis, Kind) + Δpose-Features (wie Box-Pipeline).
4. **Heads:** die **4 Targets** (gap, overlap, thickness, tipping) + `Assembly_Good` — statt der Box-Heads (overlap/thickness/gap). Multi-Label.
5. **Training** auf den Sim-Labels, dann **scale-Repair** (Größe+Position, `z` fix) gegen den GNN.

Referenz-Box-Pipeline: `src/shape_gnn_dataset.py` (Graph-Bau), `src/gnn.py` (Arch), `src/gnn_training.py`.
