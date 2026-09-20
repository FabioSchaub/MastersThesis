# Experiment 03 — Least Volume latent (form + size in one z)

**Branch:** `feature/ae-least-volume` (von `feature/ae-canonical-norm`) · **Stand: 2026-06-29**
Ergänzt `docs/SESSION_HANDOFF.md` (General-Shape-Gesamtstand).
Offizielle Referenz-Implementierung geklont nach `../../../least_volume_iclr2024/` (Repo `IDEALLab/least_volume_iclr2024`).

---

## Ziel

Das **latent-Form-Repair für General Shapes scheitert** (`SESSION_HANDOFF.md`, Phase 2): der 16-dim Form-Latent
ist **nicht bijektiv** — der Repair-Optimizer flüchtet in Off-Manifold-Bluff-`z`, die das GNN täuschen.
Idee (Chen & Fuge, ICLR 2024 — *Compressing Latent Space via Least Volume*): den Latent auf seine
**intrinsische Dimension komprimieren** (wenige aktive σ, Rest → 0) → weniger Off-Manifold-Bluff-Raum →
latent-Repair wieder möglich. Endziel: **Größe + Form in EINEM z** (Log-Scale-Kanal `z = [z_form ⊕ s]`,
`s = norm. log(max-extent)`), GNN-Knoten dann `[z | type]` statt `[z | bbox | type]`.

---

## Korrektes Verständnis (teuer erarbeitet — siehe „Reise" unten)

Least Volume = **zwei** Zutaten, und beide gehören in einen **echten Autoencoder** (NICHT den DeepSDF-Auto-Decoder):
1. **Volume-Penalty** `exp(mean(log(σ_i + η)))` auf den **Encoder-Outputs** `z = encoder(x)` — ein **WINZIGER**
   Strafterm (offiziell λ~1e-4…1e-3) über **VIELE** Epochen (offiziell bis 10000), feste LR, kein Scheduler/Warmup.
2. **Lipschitz-beschränkter (spectral-normierter) DECODER** — *die* Anti-Degenerations-Zutat. Ohne ihn „cheatet"
   das System: der Encoder squasht alle z zusammen (σ klein → Penalty klein), der ungebundene Decoder verstärkt
   die winzigen Unterschiede wieder zurück → `cos_mean → 1` (Codes fast identisch), Kompression nur scheinbar.
   Der gebundene Decoder *kann nicht* verstärken → die recon zwingt den Encoder, die Codes gespreizt zu halten
   und nur **echt ungenutzte** Dims auf 0 zu setzen.

Im offiziellen Repo wird **nur der Decoder** spectral-normiert (Encoder = normaler Discriminator). FC-Layer →
PyTorch `spectral_norm`; Conv-Layer → ihr `spectral_norm_conv`. Unser Decoder ist ein FC-MLP → Standard-`spectral_norm` reicht.

**Warum NICHT der Auto-Decoder (Stage 1):** dort sind die Codes freie Parameter ohne Encoder-Anker. Spectral Norm
beraubt sie des Gradienten (z-Starvation) → `z → 0`. Daher lebt LV bei uns in **Stage 3** (gemeinsamer
Encoder→z→Decoder-Fine-Tune), warm-gestartet vom trainierten Zwei-Stufen-Codec.

---

## Was implementiert ist

| Artefakt | Zweck |
|---|---|
| `src/least_volume.py` | `volume_penalty(z, η)`, `sigma_spectrum`, `active_dims(z, frac)`, `apply_spectral_norm`. |
| `src/dec_sdf.py` | `SDFDecoder(spectral=…, lipschitz_k=K)` (FC+ReLU+spectral_norm, Output×K) + Helfer **`sdf_decoder_from_ckpt`** (liest `decoder_spectral`/`lipschitz_k` aus dem ckpt). |
| `src/enc_pointnet.py` | `Autoencoder(spectral=…)` + Helfer **`pointnet_encoder_from_ckpt`** (liest `encoder_spectral`). |
| `src/enc_dec_training.py` | **LV in `train_stage3`** (Volume-Penalty auf `encoder(batch)`-z, konstante LR, Best-Checkpoint auf vollem Objektiv recon+λ·vol, σ-Diagnostik). **`LV_S3_ONLY`**: lädt trainierten Codec, fährt NUR Stage-3 LV (schnelles Tunen). `LV_SPECTRAL=1` im S3-only baut einen **frischen spektralen Decoder**. Stage-1-LV-Pfad existiert, ist aber dormant (λ_vol→0 an Stage 1). Checkpoints speichern `decoder_spectral`/`encoder_spectral`/`lipschitz_k`. |
| `tools/lv_warmstart.py` | Standalone Warm-Start-Fine-Tune (das 400-Shape-Free-Code-Experiment; **dessen active10=6 war ein Free-Code-Artefakt**, s.u.). |
| `autoencoder_general.slurm` | Default jetzt: **S3-only, λ_vol=1e-3, η=0.1, 300 Epochen, SN aus**; `--exclude=eu-lo-g2-028`; CUDA-Diagnose im Log. |

**Alle General-Codec-Ladestellen** (GNN-Datensatz, Repair, validate, recon-viz, check_encoder_range, lv_warmstart)
nutzen jetzt die `*_from_ckpt`-Helfer → der spektrale Codec lädt überall korrekt, Baseline-Checkpoints unverändert.

**Env-Knöpfe:** `LV_LAMBDA_VOL`, `LV_VOL_ETA`, `LV_SPECTRAL`, `LV_ENC_SPECTRAL`, `LV_LIPSCHITZ_K`, `LV_S3_ONLY`,
`LV_S3_CKPT`, `LV_WARMSTART`, `AE_N_SHAPES`, `AE_STAGE1_EPOCHS`/`AE_STAGE2_EPOCHS`/`AE_STAGE3_EPOCHS`, `AE_STAGE1_LR`.
**Checkpoints:** Baseline `best_encoder_decoder_general_canon_latentdim16.pth` (unberührt); LV → `*_general_canon_lv_*`.

---

## Stand der Ergebnisse

**recon ist über ALLE Experimente exzellent (~0.0007–0.0013, sub-mm).** Die Codec-Qualität ist nie das Problem —
nur die **Komprimierbarkeit ohne Degeneration**.

### Stage-3 LV OHNE Spectral Norm — kompletter λ-Sweep (8k Shapes, E300, `output*.log` im Masterarbeit-Ordner)
λ inferiert aus dem vol-Beitrag (train−recon):

| λ (≈) | recon | active10 | **cos mean** | |
|---|---|---|---|---|
| 0.003 | 0.00068 | **16** | 0.11 | sauber divers, **0 Kompression** |
| 0.01 | 0.00071 | 14 | 0.62 | 2 Dims, cos schon hoch |
| 0.03 | 0.00080 | 13 | 0.95 | degeneriert |
| 0.1 | 0.00102 | 13 | 0.99 | degeneriert |
| 0.2 | 0.00133 | 10 | 0.997 | total degeneriert |

→ **Definitiv: ohne Spectral Norm gibt es keinen brauchbaren Arbeitspunkt.** Entweder keine Kompression (kleines λ)
oder degenerierte Codes (cos→1). Genau der Squash-und-Verstärk-Cheat des ungebundenen Decoders.

### OFFENES, ENTSCHEIDENDES Experiment — Stage-3 LV MIT spektralem Decoder
Erstmals das **vollständige Paper-Rezept** (true-AE + Volume-Penalty + Lipschitz-Decoder) zusammen. Befehl:
```bash
for L in 0.01 0.05 0.1; do
  LV_S3_ONLY=1 LV_SPECTRAL=1 LV_LIPSCHITZ_K=6 LV_LAMBDA_VOL=$L \
    AE_N_SHAPES=8000 AE_STAGE3_EPOCHS=400 \
    sbatch --export=ALL --job-name=s3sn_$L autoencoder_general.slurm
done
```
**Schlüsselfrage:** bleibt `cos mean` NIEDRIG (<0.3) während `active10` fällt und `recon` auf ~0.005–0.02 runterkommt?
- **Ja** → Spectral Norm bricht den Trade-off, Rezept gefunden (Arbeitspunkt = größtes λ mit niedrigem cos).
- recon hängt hoch (>0.03) → K=6 zu eng für den frischen Decoder → K=10–15 oder recon-Warmup.
- cos steigt **trotz** SN → unser Form-Latent komprimiert real nicht (ehrliches Negativergebnis, sauber gezeigt).

---

## Die Reise (Sackgassen — nicht wiederholen)

1. **`lv_warmstart.py`-Sweep (400 Shapes, freie Codes, λ=0.1): active10=6** — sah nach dem Ziel aus, war aber ein
   **Free-Code-Artefakt** bei kleiner Datenmenge; überträgt sich NICHT auf encoder-z bei 8k/50k.
2. **LV im Stage-1 Auto-Decoder:** λ=0-Sanity (4675633) zeigte, dass das From-scratch-Mini-Training gar nicht
   rekonstruiert (recon 0.06, predict-mean) — Warm-Start nötig. **Spectral Norm im Stage-1** → z-Starvation,
   `z_norm 0.000` (K-Check 4747050/57/60, alle K). Stage-1 ist der falsche Ort für LV.
3. **Stage-1-Stop-/LR-Fallen:** recon-Hard-Stop feuerte bei Epoche 22; ReduceLROnPlateau würgte LR auf 1e-6
   (las den rampenden Loss als Plateau); konstante LR ließ z bei 50k *wachsen* statt komprimieren. Alle behoben,
   aber der eigentliche Fehler war: **LV gehört nicht in den Auto-Decoder.**
4. **λ zu groß / zu kurz:** wir fuhren λ=0.1 über 200 Epochen (starker Strafterm, wenige Epochen) statt des
   Paper-Regimes (winziges λ, viele Epochen). Beides falsch kalibriert.

---

## Nächste Schritte

1. **Spectral-Sweep auswerten** (s.o.) → entscheidet, ob LV bei uns trägt + (λ, K) fixieren.
2. Falls Erfolg: **voller 50k-Lauf** (`LV_S3_ONLY=1 LV_SPECTRAL=1 LV_LAMBDA_VOL=<λ> LV_LIPSCHITZ_K=<K> sbatch autoencoder_general.slurm`).
3. **Log-Scale-Kanal** (Größe in z) einbauen, Bluff-Probe darauf.
4. **GNN-Datensatz neu** (`data/shape_graphs/` leeren) → **GNN-Re-Train** mit `[z | type]` → **Repair messen**
   (`tools/shape_repair_stats.py`): sinkt die Bluff-Rate, steigt feasible über die 46% des scale-Modus?
5. Falls Negativergebnis (LV komprimiert nicht): beim **scale-Modus** bleiben, LV als verteidigbares Negativ-/Nuancen-Resultat dokumentieren.

---

## Teuer erkaufte Lektionen

1. **LV gehört in einen ECHTEN Autoencoder** (Encoder verankert z), nicht in den DeepSDF-Auto-Decoder (freie Codes → Starvation/Degeneration).
2. **Ohne Lipschitz-Decoder komprimiert LV nur via Code-Degeneration** (cos→1, squash+amplify) — kein brauchbarer Arbeitspunkt. Der spektrale Decoder ist *die* Zutat, nicht optional.
3. **Spectral Norm: im Auto-Decoder schädlich (z-Starvation), im true-AE essenziell.** Gleiche Technik, gegensätzliche Wirkung je nach Setup.
4. **Winziges λ über viele Epochen** (Paper), nicht starkes λ über wenige. Aber: unsere App unterscheidet sich (SDF/clamped-L1, FC-Decoder, andere z-Skala) → λ/K/η empirisch bestimmen, Paper-Zahlen nicht blind übernehmen.
5. **recon war nie das Problem** (~0.0007 überall) — nur Kompression-ohne-Degeneration. `cos_mean` ist die Metrik, die den Erfolg entscheidet (nicht recon, nicht active10 allein).
6. **`active10=6` (Free-Code-Sweep) war nicht repräsentativ** — bei encoder-z auf voller Skala bleibt active10 ohne SN bei 13–16.
7. **GPU:** `env_setup.sh` Lmod-Init (Commit 05a721c) + `--exclude=eu-lo-g2-028` (kaputter Node) + harter `DEVICE=="cuda"`-Assert → kein stiller CPU-Fallback.

---

## Referenz
- Paper/Repo: Chen & Fuge, ICLR 2024. PDF `../../../2783_Compressing_Latent_Space_.pdf`; Code `../../../least_volume_iclr2024/` (Kern: `src/model/sparsity.py` Volume-Penalty, `src/model/autoencoder.py` fit-Loop = festes λ, kein Scheduler/Warmup, true-AE).
- Branch `feature/ae-least-volume`, gepusht (zuletzt `28c221c`).
