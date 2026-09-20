# Session-Handoff — General-Shape-Erweiterung

> **⏩ AKTUELLE RICHTUNG (Stand 2026-06-29): Generalisierung — 14-Shape-Vokabular + reichere Targets**, Branch `feature/ae-shape-vocab-14`. Dieses Dokument hält die *Gemini*-General-Shape-Phase fest (Vorgeschichte + funktionierende Box-/Gemini-Basis). Aktueller Stand, Shape-/Target-Entscheidungen und Codec-Re-Train: **`docs/experiments/04-vocab-and-targets.md`**.

**Stand: 2026-06-17** · Branch `feature/ae-canonical-norm` (von `feature/ae-gemini-shape-family` von `feature/ae-shape-accuracy`)
Ergänzt `docs/experiments/02-shape-extension-plan.md` (Roadmap).

---

## TL;DR

Masterthesis-Pipeline: Block-Assemblies → GNN prüft Feasibility → Gradienten-„Repair" zu feasiblen Designs. Erweiterung von **box-only** auf **General Shapes** (cylinder, L-/T-Profile), getrieben von **Gemini-Shape-Mix-Figuren**.

**Encoder ✅, Datensatz ✅, Shape-GNN ✅, Repair gemessen.** Die ganze General-Shape-Pipeline steht und ist Ende-zu-Ende durchgemessen:
- Canon-z16-Encoder: alle 4 Shapes **sub-mm**.
- Shape-GNN (node21) auf 500k Paaren: **thr_F1 ~0.91, bin_F1 ~0.915**.
- **Repair: scale-Modus funktioniert (46% GT-feasible). latent-Modus (Form-Morphing) scheitert (42%/21%) an Surrogat-Exploitation** — sauberes, verteidigbares (Negativ-)Ergebnis.

**Nächster Schritt:** WS-C (Loader Gemini-Figur → n-Block-Graph) → WS-D/E (Repair + Eval box-only vs. general-shape auf echten Figuren) — mit scale-Modus.

---

## Gesamtstand (Pipeline)

| Stufe | Status | Artefakt / Ort |
|---|---|---|
| Shape-SDFs (Gemini-Konvention) | ✅ verifiziert 0.0000 mm auf 901 Blöcken | `src/shape_primitives.py`, `tools/verify_gemini_shape_sdf.py` |
| Encoder/Decoder (Canon-z16) | ✅ alle Shapes sub-mm | `encoder_decoder_model/best_encoder_decoder_general_canon_latentdim16.pth` |
| Datensatz-Gen (slender, 500k) | ✅ Gemini-Verteilung, node21 | `src/shape_assembly.py`, `src/shape_gnn_dataset.py`, `shape_dataset_gen.slurm` |
| Shape-GNN (node21) | ✅ trainiert, konvergiert | `gnn_models/gnn_shape_node21_1781643732.pth` |
| Repair scale-Modus | ✅ funktioniert (46%) | `src/shape_repair_optimizer.py` (`repair_shape_pair`) |
| Repair latent-Modus | ❌ scheitert (Negativergebnis) | `repair_shape_pair_latent` |
| Visualisierung / Dashboards | ✅ 3 Dash-Apps | `tools/shape_pair_viewer.py`, `…_repair_viewer.py`, `…_smooth_compare.py` |
| WS-C Loader Gemini→Graph | ⬜ offen | — |
| WS-D/E Repair + Eval auf Figuren | ⬜ offen | — |

---

## Ergebnisse (konsolidiert)

### Encoder-Validierung (`src/validate_shape_encoder.py`, mean\|SDF\| an der Oberfläche)

| Shape | z8 (Anfang) | WS-B (Fix 1) | **Canon (Fix 2, Referenz)** |
|---|---|---|---|
| box | 0.23–0.34 | 0.44–0.57 mm | 0.50–0.87 mm |
| cylinder | ~1.0–1.6 | 0.25–1.2 mm | **0.15–0.25 mm** |
| lprofile | ~5 | ~3.5 mm | **~0.44 mm** |
| tprofile | ~6 | ~2.5–3.2 mm | **~0.4–0.54 mm** |

Validiert auf car / shelf_rack (24× lprofile) / workbench (tprofile) / castle / cactus. HTML unter `results/shape_validation/`.

### Shape-GNN-Training (500k Paare, 80 Epochen, val)

**thr_F1 ~0.91 · bin_F1 ~0.915 · MAE_ov 1.6 mm · MAE_th 2.9 mm** (ep 60–80 flach, kein Collapse). node_dim 21 = z16 + bbox3 + node_type2.

### Repair (`tools/shape_repair_stats.py`, infeasible Paare, Seeds ≥ 2_000_000 ungesehen)

GT = analytisch (`src.shape_metrics.pair_metrics` auf der wahren SDF-Geometrie), GNN = differenzierbarer Surrogat. Bluff = GNN sagt feasible, Geometrie nicht.

| Metrik | **scale (v0)** | latent naiv | latent box-style |
|---|---|---|---|
| Repair-Erfolg (GT-feasible) | **46%** (N=100) | 42% (N=50) | 21% (N=24) |
| Bluffs | 9% | 54% | 62% |
| GNN−GT MAE overlap/thickness | 3.1 / 3.3 mm | 8.9 / 9.6 mm | 7.4 / 4.1 mm |
| Recall (GNN) | 87% | 100% | 100% |

- **scale (Größe+Position, Form fix):** 46%. Fehler zu ~94% **overlap-limitiert** → uniform-Skalierung kann overlap/thickness nicht entkoppeln. Pro Typ: box 68% › lprofile 58% › cylinder 45% › tprofile 26%. GNN gut kalibriert (Median-MAE sub-2 mm, 9% Bluffs).
- **latent (zusätzlich `z`/Form):** scheitert. Der Optimizer findet **adversariale Latents**, die den GNN täuschen (Recall 100%, Prediction von der Geometrie entkoppelt). Siehe „Phase 2" unten für das Warum.

---

## Wie wir hierher kamen (kondensiert)

### Encoder — zwei Fixes (von ~5–6 mm L/T auf sub-mm)
- **Fix 1 — WS-B: Trainings-Geometrie an Gemini angleichen** (`8dcb64b`). Neue analytische SDFs in Gemini-Konvention (Querschnitt Y-Z extrudiert entlang **X**, eine Wandstärke `t`, zentriert — `src/shape_primitives.py`). Vokabular `[box, cylinder, lprofile, tprofile]`, Gewichte `[0.30, 0.25, 0.225, 0.225]`. Verify-Gate gegen 901 echte Blöcke = 0.0000 mm. → L/T ~2.5–3.5 mm.
- **Fix 2 — Canonical Normalization** (`645a8ea`). Training fütterte rohe Wolken, Inferenz normalisiert auf max-extent 0.9 → Frame-Mismatch, traf L/T am härtesten. `CANON_EXTENT=0.9` + `canonical`-Flag, env `AE_CANON_NORM=1`. → alle Shapes sub-mm.

### Datensatz — Gemini-Verteilung + 500k (`763147a`, `641f266`)
- **`slender`-Flag** (`generate_random_shape`): balken-/platten-artiger Aspect (lange Achse + kleiner Querschnitt, Seite je `frac()` ∈ [0.12,0.45]). Echte Gemini-Blöcke sind 92% balken-artig (cross/long ~0.24); kubischer Default-Tail gekappt (thin/long p90 0.55→0.33), Zylinder verschlankt (0.49→0.29). AE behält den breiteren Default (Encoder eingefroren, slender ⊂ Default).
- **Per-Achse-Kontakt-Offset** (`assemble_pair`): tangentialer Versatz aus der echten Standfläche statt aus dem Längsachsen-Radius → gap-Rejection 19%→0.5%. `rot_a`/`rot_b` für die echte Welt-Pose gespeichert.
- **`assemble_pair` als gemeinsame Geometrie-Quelle** (`make_assembly_pair` + Repair-Test teilen sie) — verhindert die Frame-Drift, die den bbox-Bug verursacht hatte.
- **OOM-Härtung Slurm:** `--mem-per-cpu=8G` + `MALLOC_ARENA_MAX=2` (glibc-Arena-Fragmentierung durch per-Paar-Transients). Array `0-99 × 5000 = 500k`, ~8 min/Shard.

### Shape-GNN — node21 auf Canon-z16
- node_dim wird **dynamisch** aus `config.autoencoder.latent_dim` gelesen (kein 8/13 mehr hartcodiert).
- **Lektion:** Training crashte zuerst an gemischten node_dims (alte z8/node13-Shards neben neuen z16/node21). **Vor jedem Re-Gen `data/shape_graphs/shard_*.pt` löschen.** Check: `Counter(g[0].x.shape[1]) == {21: 100}`.

### Repair — Phase 0 → 2
- **Phase 0 — Portierung** (`008f0e2`): Repair auf z16/Canon-GNN umgestellt; **bbox-Frame-Fix** (GNN-bbox = `\|R\|@ptp`, flat-Pose; Optimizer baute roh → falsche Features). `[sanity]`-Check matcht jetzt exakt (3.7e-9).
- **Phase 1 — scale gemessen:** 46% (s.o.).
- **Phase 2 — latent (Negativergebnis):** s.u.

---

## Repair Phase 2 — warum latent scheitert (Kern-Lektion)

`repair_shape_pair_latent` optimiert zusätzlich `z` (Form). Zwei Bugs zuerst gefixt (`55640fa`): **Canonical-Frame** (durchgängig kanonisch parametrisiert, `s` = canonical→world; vorher dekodierte Geometrie ~2× zu klein vs. bbox → Crashes) und **degenerierte Decodes** abgefangen (`place_in_contact`-Guard + `m_after=None`).

**Befund:** Der Optimizer beutet den GNN-Surrogat aus — sobald `z` frei morpht, ist die GNN-Vorhersage von der echten Geometrie entkoppelt (Recall 100%, 54% Bluffs, ~9 mm Gap).

**Box-Mechanismen 1:1 gespiegelt (`76de621`) — hilft NICHT** (sogar schlechter, 21%): häufige Mannigfaltigkeits-Projektion (resync alle 3 statt 50: decode→sample→re-encode + bbox neu) + Hard-Caps (scale-Box + ±20mm pos-Box).

**Warum die Box (Latent 88%) gewinnt und General scheitert:**
- **Box:** Knoten = `[z, type]`, `z = encode_size(half_extents)` — **z↔Größe bijektiv**. Projektion `decode→clamp(size)→encode` ist *sauber* (jeder Latent → gültige Box), per JEDEM Schritt. GNN dicht über den ganzen glatten 8-dim-Raum trainiert → überall treu.
- **General:** Knoten = `[z, bbox, type]`, `z = encode(canonical_surface)` = **Form**, `bbox` = separate Größe → entkoppelt. `decode→SDF→sample→encode` ist verrauscht und kann **degenerieren** (die meisten ℝ¹⁶-Punkte → Müll-SDF). GNN nur auf der gesampelten Encoder-Mannigfaltigkeit trainiert → gemorphtes `z` extrapoliert.

→ **Die Box-Robustheit lebt von der bijektiven Größen-Kodierung; die fehlt dem General-Form-Latent strukturell und lässt sich nicht nachrüsten.**

**FAZIT:** scale-Modus (46%) ist das funktionierende General-Shape-Repair. Verteidigbarer Thesis-Beitrag: *Größen-Repair funktioniert; Form-Repair über den Latent scheitert, weil es — anders als der Box-Codec — keine saubere Projektion auf eine gültige Form-Mannigfaltigkeit gibt.*

**Einzige verbleibende Option für Form-Repair** (nur falls zwingend gebraucht): GNN mit **off-manifold-`z`-Augmentation** neu trainieren (Datensatz neu + Re-Train Euler), großer Aufwand, unsicher.

---

## Trainierte Modelle

- **Encoder/Decoder (REFERENZ):** `encoder_decoder_model/best_encoder_decoder_general_canon_latentdim16.pth` (ts 20260615-140441). `latent_dim 16, hidden_dim 512, ReLU, clamp_delta 0.1, batch_size 64`, kanonisch (`AE_CANON_NORM=1`). z_norm ~0.85, cos ~0.07.
- **Encoder/Decoder (Baseline ohne Canon):** `best_encoder_decoder_general_latentdim16.pth` (ts 20260615-101209).
- **Shape-GNN (REFERENZ):** `gnn_models/gnn_shape_node21_1781643732.pth` (node21, 500k, 80 Ep). **Liegt auf Euler-Scratch → wegsichern.** (Alt `gnn_shape_node13_*.pth` = obsolet.)

---

## Teuer erkaufte Lektionen (NICHT wiederholen)

1. **Canonical-Frame-Konsistenz:** Jede Stelle, die den Canon-Encoder füttert, muss vorher zentrieren + auf max-extent 0.9 skalieren (`to_canonical`). Dominierender L/T-Blocker — nicht Kapazität. Gilt auch im Repair (bbox = `\|R\|@ptp`, flat-Pose).
2. **L/T nicht durch Achsen-Korrektur retten** — es war Verteilung (Fix 1) + Frame (Fix 2).
3. **SIREN kollabiert** bei `stage1_lr 1e-3`. Nur mit lr~1e-4 + omega/warmup separat.
4. **`sdf_clamp_delta 0.04` → Latent-Collapse.** Bei 0.1 bleiben.
5. **Stage-1-`backward` OOM** → `batch_size 64`. Vor jedem AE-Submit `tools/preflight_autoencoder.py`.
6. **Daten-Gen OOM** bei 4G/cpu → `8G` + `MALLOC_ARENA_MAX=2`.
7. **Vor jedem Datensatz-Re-Gen `data/shape_graphs/` leeren** (sonst gemischte node_dims → Trainings-Crash).
8. **Nicht mehrere Hebel gleichzeitig ändern** — jeder Fix einzeln, damit der Beitrag zuordenbar bleibt.
9. **Form-Repair-Surrogat-Exploitation:** Optimieren gegen einen GNN, der nur nahe der Encoder-Mannigfaltigkeit treu ist, findet adversariale Latents. Box-Projektion rettet nur dank bijektiver Größen-Kodierung.

---

## Visualisierung / Dashboards (alle lokal, Decoder + GNN-Checkpoints müssen in `…model/` liegen)

- **`tools/shape_pair_viewer.py`** (Port 8050): blättert durch synthetische Paare (reproduzierbar pro Index, lokal regeneriert — NICHT die Euler-Shards). Zeigt die **dekodierte Rekonstruktion** (encode→z→SDF-decode→Marching-Cubes, world-posiert) + GT-Labels + **GNN-Prediction** (overlap/thickness/feasible vs. GT).
- **`tools/shape_repair_viewer.py`** (Port 8052): Repair **vorher/nachher** pro Datapoint (scale-Modus), dekodierte Meshes + GT/GNN-Metriken nebeneinander; Index zeigt auf infeasible Paare.
- **`tools/shape_smooth_compare.py`** (Port 8051): Vorher/Nachher Mesh-Smoothing (Stärke-Slider). **Smoothing bewusst NICHT im Haupt-Viewer aktiv** (Entscheidung Fabio) — `smooth_taubin(verts, faces, iters=~10)` existiert im Viewer, bei Bedarf in `_decoded_mesh`/`_shape_traces` einhängen.

---

## Werkzeuge / Befehle

```powershell
$PY = "$env:USERPROFILE\anaconda3\envs\MasterThesis\python.exe"

# SDFs gegen alle 901 echten Gemini-Bloecke pruefen
& $PY "tools\verify_gemini_shape_sdf.py"
# Canon-Encoder auf Gemini-Figur validieren
& $PY "src\validate_shape_encoder.py" --checkpoint "encoder_decoder_model\best_encoder_decoder_general_canon_latentdim16.pth" --figure <figure_dir> --candidate <candidate_xx> --no_html
# PFLICHT vor jedem Autoencoder-Euler-Submit
& $PY "tools\preflight_autoencoder.py"
# Repair-Qualitaet messen (scale | latent)
$env:SHAPE_STATS_MODE="scale"; $env:SHAPE_STATS_N="100"; & $PY "tools\shape_repair_stats.py"
```

**Slurm-Pipeline (Euler, Reihenfolge zwingend — jede Encoder/Verteilungs-Änderung invalidiert das GNN):**
`data/shape_graphs/` leeren → `sbatch shape_dataset_gen.slurm` (500k, Array 0-99) → `sbatch shape_gnn_train.slurm` (node21).

Euler-Checkpoint holen:
```bash
scp <username>@euler.ethz.ch:/cluster/scratch/<username>/MasterThesis/gnn_models/gnn_shape_node21_1781643732.pth "<lokal>/Code/gnn_models/"
```

---

## Schlüssel-Dateien

| Zweck | Datei |
|---|---|
| Shape-Vokabular + Gewichte + `slender` | `src/enc_dec_dataset_generation.py` |
| Gemini-Konvention-SDFs | `src/shape_primitives.py` |
| Assembly-Geometrie (Quelle der Wahrheit) | `src/shape_assembly.py` (`assemble_pair`) |
| Pair → GNN-Graph (Canon-Encode + bbox) | `src/shape_gnn_dataset.py` |
| Repair-Optimizer (scale + latent) | `src/shape_repair_optimizer.py` |
| Repair-Stats (scale/latent, GNN vs GT) | `tools/shape_repair_stats.py` |
| Repair-Test (per-Paar-Tabelle) | `tools/shape_repair_test.py` |
| Encoder-Validierung | `src/validate_shape_encoder.py` |
| Verify-Gate SDFs↔Gemini | `tools/verify_gemini_shape_sdf.py` |
| Dashboards | `tools/shape_pair_viewer.py` / `…_repair_viewer.py` / `…_smooth_compare.py` |
| Roadmap (WS-A..E) | `docs/experiments/02-shape-extension-plan.md` |

---

## Nächster Schritt — WS-C/WS-D/E

Encoder ✅ Datensatz ✅ GNN ✅ Repair-scale ✅. Offen:

1. **WS-C — Loader Gemini-Figur → n-Block-Graph:** CSV/JSON + `.npy`-Punktwolken → kanonisch normalisieren → Canon-z16 encodieren → Knoten (z+bbox+type) + Edges aus der `ConnectsTo`-Adjazenz.
2. **WS-D — sequentieller n-Block-Repair** auf echten Figuren mit dem **scale-Modus** (latent ist als scheiternd dokumentiert).
3. **WS-E — Evaluation box-only vs. general-shape** (= der Thesis-Beitrag).

---

## Mid-term presentation (Stand 2026-06-24)

**Datei:** `Midterm_Presentation_Fabio.pptx` (Repo-Root `Masterarbeit/`), 26 Folien, Englisch.
**Reproduzierbar:** `Code/tools/make_slide_assets.py` (alle PNG-Assets) → dann
`Code/tools/make_midterm_slides.py` (Deck). Beide untracked.

**Struktur:** Title · Motivation · Timeline (März→CoRL→jetzt Juni→Sept) · CoRL-Kontext
(+Roboterfoto) · Why GNN+repair (overlap/thickness) · **Part 1 Parameter** (in CoRL)
· „what we enhance next" · **Part 2 Latent (Boxen)** · **Part 3 Latent (4 Shapes)** ·
Status/Next/Summary. „Part" statt „Block". Pro Part: **Encoder/Decoder → GNN → Repair → Result**.
Verbindender Faden: gleiche GNN+Repair, node-Repräsentation entwickelt sich
([size|type]→[z|type]→[z|bbox|type]).

**Assets/Render-Tech:**
- Assemblies via **trimesh offscreen (echter Z-Buffer)** — matplotlib-3D verworfen (falsche Verdeckung). Helper `_tm_image`/`_cam_xform` in make_slide_assets.
- GNN-Figuren `gat_part1/2/3.png` = matplotlib-Nachbau der Paper-TikZ (`gnn.txt`), Input 5/10/21.
- E/D-Figuren `encdec_part2.png` (Box-Codec 3→128→128→128→8, bijektiv) / `encdec_part3.png` (PointNet+SDF).
- Repair-Loop 5-Schritt-Diagramm (`repair_loop`), pro Part wiederverwendet.
- `gt_vs_recon.png` aus den STL-Drucken (`results/stl_prints/`, 40 mm, GT vs recon).
- `assemblies_gallery.png`, `repair_before_after.png` (cactus) aus Pipeline-CSVs.
- **Folie 16** = `fig16_illustration()` → `fig16_chair.png`: hand-gebaute SCHEMATISCHE
  Illustration (NICHT echter Repair): Input mit dicken grün+lila Rückenlehnen-Posten →
  Parameter dünnt sie → Latent dünnt sie etwas anders (Latent ≠ Parameter, klein).
  Explizite Farben, ganzer Stuhl sichtbar.

**Offen:** canonical Latent-Checkpoint für Deck-Zahlen festlegen (131321 = Deck-88 % vs
124934 = gemessene 83/84 %). 3 Bild-Platzhalter sind ersetzt. Optional Paper-Fig.1 einbetten.
