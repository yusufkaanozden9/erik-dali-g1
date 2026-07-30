# Reference motion source

- **File**: `reference_motion/raw/erik_dali_reference.mp4` — user-supplied
  (`erik dalı videosu.mp4` from `~/Downloads`). Used as-is, no trimming needed.
- **Properties**: 23.67s, 1568x1770, 30fps.

Note: an earlier version of this file used a 4:53–5:16 trim from a different, longer
source video (`ERİK DALI NASIL OYNANIR - 5 Dakikada İzle Öğren.mp4`, same dancer/studio
set). The user replaced it with this dedicated clip — this is now the canonical reference
for the pipeline.

## Visual sanity check (done)

Sampled 5 frames across the clip and inspected them directly:

- Single dancer, centered, facing the camera the whole time.
- Camera is static (same studio/stage set as before — fixed shot, red curtain background).
- Full body incl. both feet visible in every sampled frame.
- Flat, matte wooden floor.
- Dancer holding kaşık (spoons) — GVHMR/SMPL-X track body pose only, not props, so the
  spoons themselves won't be retargeted, but the arm/wrist motion driving them will be.
- More dynamic than the earlier trim: includes footwork/turns and some fast arm raises —
  a couple of sampled frames show motion blur on the moving arm/foot. Fine for a first
  retargeting pass; if GVHMR's output looks noisy on the fast segments specifically, that's
  the likely cause (worth a visual check after stage 1, not just stage 4).

## Caveats (per the PDF's own guidance)

- This is a **video-based pose estimate** (GVHMR), not real motion-capture (Xsens/OptiTrack).
  Expect noisier retargeting than a clean mocap source.
- Camera is static → `scripts/01_extract_smplx.sh` runs GVHMR with `-s` (skip visual
  odometry), which is correct for this clip. If a future replacement clip has camera
  motion, drop that flag.
