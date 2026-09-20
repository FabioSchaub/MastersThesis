# Experiment-Plan 02 — Erweiterung auf General Shapes (Gemini-Shape-Mix)

**Datum**: 2026-06-14
**Ausgangs-Branch**: `feature/shape-extension`
**Constraint**: shape-agnostisch — Repair optimiert immer nur über `z` (Latent)
und `pos`. Keine box-spezifischen Features.

Ziel: Die box-only Repair-Pipeline so erweitern, dass sie die mit Gemini
generierten **Shape-Mix-Figuren** (box / cylinder / lprofile / tprofile aus
`Woodworking_Simulation/scripts/shape_dataset/`) end-to-end durch GNN + Repair
verarbeitet — und nicht nur deren Bounding-Box.

---

## 1. Verifizierter Stand (was schon existiert)

**Code-Seite (`Code/`)**
- General PointNet-Autoencoder **trainiert**:
  `encoder_decoder_model/best_encoder_decoder_general_latentdim8.pth` (2.1 MB).
- Shape-Vokabular: `src/enc_dec_dataset_generation.py:535`
  `SHAPE_TYPES = ["box", "cylinder", "rounded_box", "lbracket", "tshape"]`
  (SDF-basiert, synthetisch generiert).
- Shape-Assembly-Paare + Graph-Dataset:
  `src/shape_assembly.py`, `src/shape_gnn_dataset.py`
  (Node = `z(8) + bbox(3) + node_type(2)` = 13 dim; paper-v0:
  Regression overlap + thickness, **kein** gap — gap wird vom Contact-Snap
  auf ~0 getrieben).
- Trainings-Infrastruktur auf Euler: `shape_dataset_gen.slurm` →
  `data/shape_graphs/shard_*.pt` → `shape_gnn_train.slurm`.

**Woodworking-Seite (`Woodworking_Simulation/`)**
- Shape-aware Gemini-Harness: `scripts/generate_shape_dataset.py`,
  `scripts/shape_dataset/{shape_library,shape_prompt,exporters}.py`.
- Pro Kandidat exportiert: `.yaml`, `.json` (SceneState), `.csv`
  (ML-Input-Schema), `pointclouds/*.npy` (analytische Oberflächen-Punktwolke
  pro Block, lokal + Welt).
- Bereits generiert: cactus, castle, car × 8 Kandidaten mit Shape-Mix
  (`structures/shape_dataset/dataset_index.json`).
- `cactus_candidate_4` und `castle_candidate_4` liegen schon in
  `Code/pipeline/new_csv/` (inkl. `_repaired.csv`).

---

## 2. Verifizierte Lücken (was die Erweiterung blockiert)

**L1 — Die CSV trägt keine Shape-Identität.**
Das ML-Input-Schema (`Block<i>_PosX/Y/Z`, `_SizeX/Y/Z`, `_ConnectsTo_`,
`_Screw<k>_`) kennt nur die **Welt-AABB**. Ein Gemini-Zylinder wird in der CSV
zu seiner Bounding-Box — `cylinder` vs `lprofile` ist nicht unterscheidbar.
Die echte Form überlebt **nur** in den `.npy`-Punktwolken.
→ Solange Repair die CSV liest, repariert es Boxen, keine Shapes.

**L2 — Repair-Loop ist nicht auf General Shapes verdrahtet.**
`src/repair_process.py`, `src/repair_optimizer.py`, `pipeline/dashboard.py`
referenzieren weder `best_encoder_decoder_general`, noch `PointNetEncoder`,
noch `shape_gnn_dataset`. Der evaluierte Repair läuft also **box-only**
(BoxEncoder-Latent). Der General-Encoder ist trainiert, aber nicht eingebunden.

**L3 — Vokabular- und Achsen-Mismatch.**
Woodworking: `lprofile` / `tprofile` / `cylinder` (lokale X = Länge).
Code: `lbracket` / `tshape` / `cylinder` (`cylinder_sdf` entlang Y-Achse).
Parametrisierung und Achsenkonvention unterscheiden sich → der PointNet sieht
sonst eine andere Geometrie, als Gemini meint.

**L4 — Kein Loader Gemini-Figur → Graph.**
`shape_gnn_dataset.py` baut Graphen aus **synthetischen** `shape_assembly`-Paaren
(2 Knoten). Es gibt keinen Pfad, der eine n-Block-Gemini-Figur (CSV + Punktwolken)
in einen Repair-tauglichen Graphen übersetzt.

---

## 3. Roadmap (in Abhängigkeitsreihenfolge)

### WS-A — Shape-Identität in den Datenfluss bringen (Voraussetzung für alles)
Die `.npy`-Punktwolken sind die Wahrheit; die CSV ist lossy. Zwei Optionen:
- **A1 (empfohlen):** Repair konsumiert pro Block die **Punktwolke → PointNet →
  `z`**, nicht die CSV-Size. Die CSV bleibt nur für Pos/Connects/Screw. Damit
  ist der Pfad nativ shape-agnostisch und nutzt den schon trainierten Encoder.
- **A2 (Fallback):** CSV-Schema um eine `Block<i>_ShapeType`-Spalte +
  Shape-Parameter erweitern. Schneller, aber bricht die „shape = Latent"-Idee
  und das bestehende Schema.
→ **Verify:** Für `cactus_candidate_4` Block-Latents aus `.npy` rekonstruieren
  (SDF-Decoder) und visuell gegen die Gemini-Geometrie prüfen.

### WS-B — Vokabular/Achsen angleichen (L3)
Mapping-Tabelle Woodworking ↔ Code festlegen (`lprofile`↔`lbracket`,
`tprofile`↔`tshape`, Zylinderachse). Entweder (a) Woodworking-`sample_surface`
erzeugt Wolken in der Code-Encoder-Konvention, oder (b) ein Adapter rotiert/
reparametrisiert beim Laden. Eine Konvention als Single Source of Truth.
→ **Verify:** Encode→Decode-Roundtrip pro Shape-Typ, Chamfer-Distanz < Schwelle.

### WS-C — Loader Gemini-Figur → n-Block-Graph (L4)
Funktion, die `<figure>/candidate_k.{csv,json}` + `pointclouds/*.npy` in die
sequentielle n-Block-Repair-Struktur (`repair_process.py:334`) übersetzt:
Knoten = `z` (aus Punktwolke) + bbox-Skalar + node_type, Edges wie gehabt.
→ **Verify:** Smoke-Test analog `tools/smoke_test_graph.py` auf einer Figur.

### WS-D — Repair-Loop auf General-Encoder umstellen (L2)
`repair_process` / `repair_optimizer` so erweitern, dass die Optimierung über
den **General-Latent** `z` + `pos` läuft (BoxEncoder als Box-Spezialfall
behalten oder ersetzen). Shape-GNN-Checkpoint (`shape_gnn_train.slurm`-Output)
statt box-GNN laden.
→ **Verify:** `tools/repair_csv_sweep.py` analog, aber auf den Shape-Figuren;
  Feasibility analytisch messen (overlap/thickness), Baseline = box-only.

### WS-E — Evaluation & Thesis-Beitrag
Head-to-Head: box-only Repair vs general-shape Repair auf demselben Figuren-Set.
Metriken: analytische Feasibility-Rate, Shape-Treue (Chamfer vor/nach Repair),
Bluff-Analyse wie im box-Fall. Das ist die eigentliche Thesis-Erweiterung.

---

## 4. Offene Entscheidungen (vor WS-A klären)
- **Gap im General-Fall:** paper-v0 lässt gap weg (Snap → 0). Bleibt das so,
  oder braucht der Shape-Fall einen echten gap-Pfad? (Beeinflusst GNN-Heads.)
- **Screws/Connects:** bleiben Screw-Positionen Repair-Input oder nur Kontext?
- **A1 vs A2** (WS-A): nativ-latent vs CSV-Spalte.
- **Vokabular-Scope:** alle 4 Woodworking-Shapes, oder erst box+cylinder als
  minimaler Erweiterungsschritt?

---

## 5. Nächster Schritt
Mit **WS-A1** starten und an **einer** Figur (`cactus_candidate_4`, bereits in
`pipeline/new_csv/`) verifizieren, dass Punktwolke → PointNet → `z` → SDF-Decode
die Gemini-Geometrie trifft. Das ist die kleinste end-to-end überprüfbare
Einheit und entscheidet A1 vs A2.
