# Reference motion source

- **Source file (user-supplied)**: `ERİK DALI NASIL OYNANIR - 5 Dakikada İzle Öğren.mp4`
  (originally downloaded by the user from YouTube into `~/Downloads`).
- **Trimmed segment**: `04:53` → `05:16` (23s), extracted with:

  ```bash
  ffmpeg -i "ERİK DALI NASIL OYNANIR - 5 Dakikada İzle Öğren.mp4" \
    -ss 00:04:53 -to 00:05:16 \
    -c:v h264_videotoolbox -b:v 6M -c:a aac \
    erik_dali_clip_04m53s-05m16s.mp4
  ```

  Output: `reference_motion/raw/erik_dali_clip_04m53s-05m16s.mp4` — 1280x720, 30fps, 23s.
  (`h264_videotoolbox` used because the local ffmpeg build has no GPL `libx264`; on the
  Linux GPU box used for the actual GVHMR run, re-encode with `libx264`/`libopenh264`
  instead — `h264_videotoolbox` is macOS-only.)

## Caveats (per the PDF's own guidance)

- This is a **video-based pose estimate** (GVHMR), not real motion-capture (Xsens/OptiTrack).
  Expect noisier retargeting than a clean mocap source.
- If the clip has camera motion, drop the `-s` (skip visual odometry) flag when running
  GVHMR's `demo.py` — see `scripts/01_extract_smplx.sh`.

## Visual sanity check (done)

Sampled 4 frames across the trimmed clip and inspected them directly — conditions match
the PDF's checklist for a good retargeting source almost exactly:

- Single dancer, centered, facing the camera the whole time.
- Camera is static (studio/stage set, fixed tripod shot).
- Full body incl. both feet visible in every sampled frame.
- Flat, matte wooden floor — no glare/reflection that would confuse foot-contact estimation.
- Plain curtain background, good separation from the subject.
- Dancer is holding kaşık (spoons) — GVHMR/SMPL-X track body pose only, not props, so the
  spoons themselves won't be retargeted, but the arm/wrist motion will be.

No re-trim needed. This clip can go straight into `scripts/01_extract_smplx.sh` once GVHMR
is installed (see `setup_envs.sh`).
