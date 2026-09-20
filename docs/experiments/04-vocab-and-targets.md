# Experiment 04 — Generalisierung: 14-Shape-Vokabular + reichere GNN-Targets

**Branch:** `feature/ae-shape-vocab-14` (von `feature/ae-least-volume`) · **Stand: 2026-07-01**
Fortsetzung von `03-least-volume-latent.md`. Ergänzt `docs/SESSION_HANDOFF.md`.

---

## TL;DR — Stand & nächster Schritt (für neue Sessions)

Codec-Phase 1 ist **abgeschlossen**. Generalisierung weg von Woodworking/Gemini; Beitrag = **Shapes +
reichere Targets** (LV/Encoder = Infrastruktur, NICHT der Beitrag).

- **Shape-Vokabular (fix):** 14 Shapes ohne Genus-1 — `box, rounded_box, cylinder, half_cylinder,
  sphere, ellipsoid, capsule, cone, hex_prism, wedge, i/u/l/t_profile`. Viewer: `tools/shape_gallery_viewer.py`.
- **Codec (FERTIG, nutzen):** LV-komprimiert, latent_dim 32 → **effektiv ~10 aktive Dims**.
  → `encoder_decoder_model/best_encoder_decoder_general_canon_lv_v14_lam005_latentdim32_active9.pth`
  (9-aktiv-Lauf). **12/14 Shapes sub-mm** (nur sphere/capsule ~1.2 mm = Smooth-Limit). Belege:
  `tools/validate_vocab_recon.py`, `tools/codec_quality_report.py`.
- **Targets (ENTSCHIEDEN):** `{overlap, thickness, stability, collision}` + `Assembly_Good`-Head.
  `stability`+`collision` sind **global** (→ Graph-Head, n-Block nötig). `stability`-Label gebaut:
  `src/shape_stability.py`. `table` = Input-Feature; `gap`/`gripper`/screwdriver-Modi raus.

**NÄCHSTER SCHRITT (Phase 2, noch nicht begonnen):** n-Block-Assembly-Generator (stability/collision
brauchen 3+ Teile) → `collision`-Label (analytisch) → GNN mit **Graph-Level-Head**, Knoten
`[z | bbox | type]` → **scale**-Repair (latent-Repair scheitert an Bluffs). PartNet-Loader (offenes WS-C)
für echte Figuren.

**Constraints:** Euler-GPU max **24 GB** (keine A100; `--gres=gpu:1,gpumem:24g`). Cluster-`sbatch`/SSH
nur durch den User. `git push` nur nach Rückfrage. Env: `C:\Users\Fabio\anaconda3\envs\MasterThesis\python.exe`.

---

## Ziel / Richtung

Generalisierung **weg von Woodworking/Gemini**. Der **Beitrag sind (1) eine reiche Shape-Familie
und (2) reichere Feasibility-Targets** über `overlap`/`thickness` hinaus. Der **LV-konditionierte
Latent + Encoder ist nur Infrastruktur (Mittel zum Zweck), NICHT der Beitrag** — explizite
Entscheidung des Users.

Killer-Demo-Ziel: Assemblies, die alle *paarweisen* Checks bestehen, aber *global* infeasibel sind
(kippen / kollidieren), von GNN + **scale-Repair** erkannt & gefixt. (latent-Repair scheitert an
Bluffs — dokumentiert im SESSION_HANDOFF; scale ist das Arbeitspferd.)

Rahmen: Budget mittel (einige Euler-Re-Trains, **kein** Isaac/FEM, analytische Labels). Datensatz
für echte Figuren später: **PartNet (Möbel)**. Repair-Modus: **scale**.

---

## Shape-Vokabular — ENTSCHIEDEN: 14 Shapes (ohne Genus-1)

`box, rounded_box, cylinder, half_cylinder, sphere, ellipsoid, capsule, cone, hex_prism, wedge,
i_profile, u_profile, l_profile, t_profile`

Begründung = **Generalitäts-Framing**: spannt die geometrischen Kategorien auf — prismatisch
(box/cylinder/hex/wedge/i/u/l/t), gekrümmt (sphere/ellipsoid/capsule/cone), gekrümmter Querschnitt
(half_cylinder). **Genus-1 (tube/torus/rect_frame) bewusst raus** (zu hohes Encoder/LV-Risiko;
evtl. später für die Isaac-Sim). `tapered_box` raus (cone deckt Verjüngung ab), `half_cylinder` rein
(füllt die Kategorie „gekrümmter Querschnitt"). Alle SDFs in Gemini-Konvention (Y-Z-Querschnitt
entlang X extrudiert, zentriert), außer sphere/ellipsoid (orientierungsfrei) und box (eigener
Face/Edge/Corner-Sampler).

---

## Targets — ENTSCHIEDEN: {overlap, thickness, stability, collision} + Assembly_Good

**Domänen-begründet aus dem Woodworking-Repo** (NICHT erfunden) — die Isaac-Sim-Output-CSV
(`Masterarbeit/num_blocks_8_*.csv`) + `rosbridge/failure_utils.py` (8 Failure-Flags) +
`dataset_generation/bloxnet_criteria.py` definieren die Feasibility.

| Target | Domänen-Quelle | Ebene | Status |
|---|---|---|---|
| **overlap** (≥10 mm beide Tangential-Achsen) | `OverlapTangential1/2Val`, `overlap_required=0.01` | paarweise | hast du |
| **thickness** (≤20 mm entlang Schraubachse) | `ThicknessMinus20mmVal`, `width<0.02` | paarweise | hast du |
| **stability** (CoM über Stützpolygon, signierte Marge) | Isaac `TIPPING` (`tilt_cos<0.995`) — analytischer Proxy | **global** | gebaut (`src/shape_stability.py`) |
| **collision** (nicht-verbundene Teile durchdringen sich) | Blox-Net Krit. 2 — NICHT in der CSV → neu labeln | **global** | offen (Phase 2) |
| Assembly_Good | UND aller Failure-Flags | global Binary-Head | hast du |

**NICHT genommen:** `table-interference` (trivial Ein-Block → bleibt Input-Feature `bottom_over_table`),
`gap` (`gap_probability=0` → kein Signal), `gripper` (im Code deaktiviert), `screwdriver-table/-block/
info-hit` (Roboter/Isaac, nicht shape-generalisierbar). `stability` + `collision` sind GLOBAL →
erzwingen einen **Graph-Level-Readout-Head** + **n-Block-Assemblies** (2-Block-Paare reichen nicht).

---

## Was implementiert ist

| Artefakt | Zweck |
|---|---|
| `tools/shape_gallery_viewer.py` | Standalone-Dash-Viewer (Port 8088) der 14 Shapes, aus analytischen SDFs gemesht. Self-contained. |
| `src/shape_primitives.py` | +9 neue SDFs (half_cylinder, capsule_x, cone, hex_prism, wedge, i/u-profile + Helfer) in X-Konvention. |
| `src/enc_dec_dataset_generation.py` | `SHAPE_TYPES`=14, Gewichte, Parameter-Sampling pro Shape (im Einheits-Maßstab innerhalb `bounds`). |
| `tools/verify_shape_vocab.py` | Offline-Gate: alle 14 SDFs gültig (mean\|SDF\|@surface, Interior, finite, in-bounds). Grün. |
| `src/shape_stability.py` | Globale Stabilitäts-Marge (CoM über Stützpolygon). Self-test auf gestapelten Boxen verifiziert. |
| `tools/validate_vocab_recon.py` | **Per-Typ-Rekonstruktions-Gate**: generiert je Typ K Instanzen, encode→decode, \|SDF\|@surface in mm. (Gemini-Figuren enthalten die neuen Shapes nicht.) |

**Config** (`config/config.yaml`, autoencoder/sdf_decoder): `latent_dim 32`, `n_shapes 100000`,
`batch_size 128`, `n_query 5000`, `lambda_vol 0.01`, `stage1_recon_threshold 0.002` (in
`enc_dec_training.py`, env `AE_S1_RECON_THRESH`). Env-Overrides: `AE_BATCH`, `AE_NQUERY`,
`AE_N_SHAPES`, `AE_RUN_TAG` (eindeutiger `.pth`-Suffix für parallele Sweeps).

---

## Stand der Ergebnisse

### Baseline v14, latent_dim 16 → FEHLGESCHLAGEN für gekrümmte Shapes

`tools/validate_vocab_recon.py` auf `…v14_latentdim16…` (per-Typ, mean\|SDF\| @135 mm-Block):

| sub-mm (OK) | mm | | FAIL | mm |
|---|---|---|---|---|
| box | 0.70 | | cone | **4.23** |
| iprofile | 0.76 | | capsule | **4.12** |
| cylinder | 0.88 | | ellipsoid | **3.06** |
| rounded_box | 0.96 | | sphere | **1.94** |
| | | | wedge / half_cylinder | 1.79 / 1.49 |
| | | | l/t/u-profile, hex | ~1.0–1.15 |

**Muster:** flächig/prismatisch ok, **gekrümmt/glatt versagt** (2–4 mm). Selbst die Kugel (1.94 mm)
schlechter als die Box (0.70). Ursachen: Codec ist box-getunt (Box hat eigenen Edge/Corner-Sampler,
PointNet lernt kanten-orientiert); `latent_dim 16` war für 4 prismatische Shapes getunt; der
Stage-1-Early-Stop bei **avg**-recon 0.005 verdeckte, dass die Kurven noch bei ~0.02 lagen.

### Baseline v2: latent_dim 32 (2026-06-29) — 10/14 sub-mm

`latent_dim 16→32` + `stage1_recon_threshold 0.005→0.002` (Stage 1 trainiert weiter; env
`AE_S1_RECON_THRESH`). → `…v14_latentdim32_20260629-215612.pth`. Per-Shape-Recon: **großer Sprung,
10/14 sub-mm** (curved von 2–4 mm auf ~1–1.3 mm), nur die 4 glatten knapp über 1 mm (sphere 1.33 /
ellipsoid 1.26 / capsule 1.20 / cone 1.02). Stage-1 recon 0.002, Stage-2 val 0.018 (d16: 0.048).
Als milde Limitation **akzeptiert** (Codec = Infrastruktur) → LV-Sweep.

### LV-Sweep (auf d32) + Per-Shape-Validierung (2026-07-01) — 9-aktiv GEWÄHLT

4× S3-only LV (frischer Spectral-Decoder K=6, N=50k, 400 Ep), λ = 0.005 / 0.01 / 0.05. σ-Cliff
monoton: active 30 → 24 → 9 → 5. **Recon bleibt flach von 30 bis 9 aktiv, bricht erst bei 5 →
intrinsische Dimension der 14er-Familie ≈ 9.**

Per-Shape-Recon (`validate_vocab_recon`, mm @135 mm) auf den 4 finalen Checkpoints:

| | 30 aktiv | 24 aktiv | **9 aktiv** | 5 aktiv (λ0.05) |
|---|---|---|---|---|
| **sub-mm** | 12/14 | 12/14 | **12/14** | 7/14 |
| sphere | 1.19 | 1.20 | 1.20 | 1.51 |
| capsule | 1.16 | 1.16 | 1.18 | 1.26 |
| cone | 0.95 | 0.92 | 0.99 | 1.18 |
| ellipsoid | 0.89 | 0.89 | 0.91 | 1.07 |

**Befunde:** (1) Kompression 30→9 kostet ~nichts an Recon (pro Shape identisch). (2) Der LV-Feintune
(400 Ep frischer Decoder) **verbessert** die Recon ggü. d32-Baseline — **cone + ellipsoid werden
sub-mm** (waren FAIL). (3) `cos 0.447` (9-aktiv) schadet der Recon NICHT (durch tote Dims aufgebläht
+ irrelevant fürs frozen-z scale-Repair). (4) λ=0.05 (5 aktiv) über-komprimiert (7/14 FAIL).

**ENTSCHEIDUNG — Codec für Phase 2 = der 9-aktiv-Checkpoint:**
`best_encoder_decoder_general_canon_lv_v14_lam005_latentdim32_active9.pth` (lokal in
`encoder_decoder_model/`). Kompakt, 12/14 sub-mm, Recon besser als der unkomprimierte Baseline. Nur
sphere/capsule (~1.2 mm) bleiben das Smooth-Shape-Limit (nachrüstbar via Encoder-Kapazität, falls je nötig).

### Codec-Qualitäts-Checks (2026-07-01, `tools/codec_quality_report.py`)

σ-Spektrum, Code-Korrelation (cos), z-Norm und **Prunbarkeit** (nur Top-k Dims behalten, Rest auf
Mittel → Recon) auf 210 synthetischen Codes:

| Codec | σ-Spektrum | cos (alle / aktiv) | z-Norm |
|---|---|---|---|
| d32 Baseline (kein LV) | flach 0.35→0.16 (unkomprimiert) | 0.013 / 0.013 | 1.31 |
| lam001 (24-ish) | 0.167→0.008 (gradueller Abfall) | 0.095 / 0.095 | 0.35 |
| **9-aktiv (lam005)** | 0.111→0.002 (weiche Knie ~9) | **0.452 / 0.452** | 0.28 |
| λ=0.05 (5-aktiv) | 4 Dims, dann 28× exakt 0 | 0.346 / **0.254** (aktiv) | 0.26 |

**Prunbarkeit** (Recon in mm, nur die Top-k σ-Dims behalten):

| behaltene Dims | lam001 | **9-aktiv** |
|---|---|---|
| 8 | 3.19 | **1.70** |
| 10 | 2.39 | **1.19** |
| 12 | 2.01 | **1.10** |
| 16 | 1.45 | 1.02 |
| 24 | 1.03 | 0.97 |
| 32 | 0.92 | 0.96 |

**Befunde:** (1) Der **9-aktiv packt die Familie in ~10–12 Dims** (Recon plateaut ab top12); lam001
verteilt die Varianz über ~24 → der 9-aktiv ist der *wirklich kompakte* LV-Codec (intended LV-Effekt).
(2) **cos 0.45 (9-aktiv) ist ECHT** — die schwachen Dims sind σ≈0.002, nicht exakt 0, daher
`cos(aktiv)=cos(alle)`. Es ist die *Signatur der Varianz-Konzentration*, KEIN Defekt: Per-Shape-Recon
12/14 sub-mm (Shapes distinkt), und für frozen-z scale-Repair irrelevant. (Nur λ=0.05 hat 28 exakt-tote
Dims → dort `cos(aktiv 4)=0.25 < cos(alle)=0.35`.) (3) **effektive intrinsische Dim ~10–12** (recon-basiert;
die frühere „~9" war die reine σ-Schwellen-Zählung).

---

## Euler-Workflow (Codec, Phase 1)

**GPU-Realität (verifiziert):** KEINE A100 verfügbar (der „A100 80GB"-Handoff ist falsch/veraltet).
Max VRAM = **24 GB** (quadro_rtx_6000 / nvidia_titan_rtx, ~17 Nodes), sonst 11 GB rtx_2080_ti. Die
Slurm fordert `#SBATCH --gres=gpu:1,gpumem:24g` + `--gpus-per-node=1`. `batch 128 × nq 5000 ≈
13–15 GB` → passt auf 24 GB. **WARNUNG:** `--gpus=1`+`gpumem:40g` routet nach `gpupr.24h` →
QOSGrpCpuLimit (Pending, läuft nie) — nicht nutzen.

1. **Baseline** (Stage 1/2, LV aus, Spectral aus — der Kapazitäts-Gate):
   ```
   sbatch --export=ALL,LV_S3_ONLY=0,LV_LAMBDA_VOL=0,AE_STAGE3_EPOCHS=0,AE_RUN_TAG=_v14 autoencoder_general.slurm
   ```
   → `best_encoder_decoder_general_canon_v14_latentdim{N}.pth`. Dann `validate_vocab_recon` → alle sub-mm?
2. **3 parallele LV-Feintunings** (erst nach gutem Baseline) — S3-only, laden den v14-Baseline,
   Spectral AN (nur hier!), distinkte Namen:
   ```
   BASE=encoder_decoder_model/best_encoder_decoder_general_canon_v14_latentdim{N}.pth
   sbatch --export=ALL,LV_S3_ONLY=1,LV_S3_CKPT=$BASE,LV_SPECTRAL=1,LV_LAMBDA_VOL=0.005,AE_RUN_TAG=_v14_lam0005 autoencoder_general.slurm
   # … 0.01 / 0.05 analog
   ```
   Danach σ-Cliff (`active`) vs Recon über die 3 λ → λ mit klarem σ-Cliff ohne Recon-Anstieg wählen.

---

## Nächste Schritte

**Codec ist DURCH** — der 9-aktiv-LV-Checkpoint steht (12/14 sub-mm, effektive Dim ~9). Offen:

1. **Phase 2 (Targets/GNN):** n-Block-Assembly-Generator (stability + collision brauchen 3+ Teile),
   `collision`-Label (analytisch), GNN mit **Graph-Level-Head** für die zwei globalen Targets,
   Knoten = `[z(9) | bbox | type]`, scale-Repair dagegen. PartNet-Loader (= das offene WS-C) für echte Figuren.
2. *Optional/später:* falls sphere/capsule (~1.2 mm) je stören → Encoder-Kapazität (`encoder_fc_dims`/
   `encoder_point_mlp_dims`) hoch + Codec neu; sonst als dokumentierte Smooth-Shape-Limitation belassen.

---

## Teuer erkaufte Lektionen (dieser Branch)

1. **Per-Typ validieren, nicht nur den avg-recon.** Stage-1-Early-Stop bei avg 0.005 sah „konvergiert"
   aus, während die halbe Familie (Kurven) bei 2–4 mm lag. `validate_vocab_recon` deckte es auf.
   Gemini-Figuren enthalten die neuen Shapes nicht → eigenes per-Typ-Gate nötig.
2. **GPU-Realität prüfen, nicht dem Handoff glauben.** „A100 80GB" existierte nicht; mehrfach OOM
   (512/256 × nq 8000) auf 11 GB/24 GB. `sacct -j <id> --format=AllocTRES` zeigt die echte Karte.
   `--gres=gpumem:24g` für die 24-GB-Karten; `gpumem:40g` gibt's hier nicht (→ gpupr/QOS-Pending).
3. **Slurm-Env überschreibt Config still.** `autoencoder_general.slurm` hatte `AE_N_SHAPES` hart
   gesetzt → überschrieb `config.n_shapes`. Jetzt nicht mehr forciert (Config = Quelle der Wahrheit).
4. **Memory ≈ batch × n_query** (eine Decoder-Layer-Aktivierung = `batch·n_query·hidden·4 B`).
   `n_shapes` ist KEIN Memory-, sondern ein Wall-Clock-Treiber.
5. **Eindeutige `.pth`-Namen via `AE_RUN_TAG`** — sonst überschreiben sich parallele Läufe (gleicher
   `name_tag`). `latentdim{N}` im Namen trennt d16/d32 automatisch.

---

## Referenz

- Paper: Chen & Fuge, *Compressing Latent Space via Least Volume*, ICLR 2024 (`../../../2783_Compressing_Latent_Space_.pdf`).
- Vorgänger: `03-least-volume-latent.md` (LV-Stand), `02-shape-extension-plan.md` (Gemini-Roadmap), `SESSION_HANDOFF.md` (General-Shape-Gesamtstand).
- Isaac-Failure-Taxonomie: `Woodworking_Simulation/.../rosbridge/failure_utils.py`, `dataset_generation/bloxnet_criteria.py`, `randomize_blocks.py`; Output-CSV `Masterarbeit/num_blocks_8_*.csv`.
