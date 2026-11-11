# mold at bf0ea4a0, with the Wan VAE tiled-decode fix

mold (https://github.com/utensils/mold), MIT (see `LICENSE`; parts of mold-inference and mold-candle are Apache-2.0
and others, see each crate's `LICENSE-*` and `THIRD_PARTY_NOTICES.md`), at rev
`bf0ea4a0b8276fb60e7ea0dbb431967f76ee035b`, the revision `crates/strata-media/Cargo.toml` pins.

Only the six crates the media engine builds are kept: mold-inference, mold-core, mold-candle, mold-catalog,
mold-secure and mold-secure-client. `crates/strata-media/Cargo.toml` points its `[patch."https://github.com/utensils/mold.git"]`
section here. The folder sits at the repository root, not under strata-media, because path dependencies inside the
strata-media folder would become members of its workspace and could not inherit mold's `workspace.package` fields.

What differs from upstream at that revision:
- **The fix**, in `crates/mold-inference/src/wan/model/vae.rs`, `WanVideoVae::decode_tiled`:

  ```diff
                   let mask = tile_blend_mask(out_h, out_w, overlap * scale, &device)?;
  -                let contribution = decoded.broadcast_mul(&mask)?;
  +                // am2Strata: the accumulators are on the CPU, so the tile's share and its mask go there too (adding
  +                // a CUDA tile to a CPU accumulator failed with "device mismatch in add")
  +                let contribution = decoded.broadcast_mul(&mask)?.to_device(&accumulator_device)?;
  +                let mask = mask.to_device(&accumulator_device)?;
  ```
- **Left out:** each crate's `testdata/`, `tests/`, `examples/` and `benches/` (not read by a normal build; a few
  `#[cfg(test)]` modules `include_bytes!` them, so mold's own unit tests do not build here), and the rest of mold's
  workspace (CLI, server, apps, web). The root `Cargo.toml` is mold's, trimmed to these six members.
- Because the copy is inside this repository's git, mold-core's build script reports this repository's commit, not
  mold's (set `MOLD_GIT_SHA` to change it).

**Why:** Wan 2.2 TI2V 5B at its native 1280×704 (121 frames) needs more than 16 GB for the whole VAE decode, so mold
retries tiled. Its tiled decode keeps the totals on the CPU on purpose but added each CUDA tile to them, and failed
with `device mismatch in add, lhs: Cpu, rhs: Cuda { gpu_id: 0 }` after the whole 36-minute render (measured on
2026-11-11, RTX 5060 Ti 16 GB). With the fix the clip finished: 2,612 s, 7 min of it the tiled decode.

**Remove it** when upstream mold fixes `WanVideoVae::decode_tiled` and strata-media moves to that revision: delete
this folder and the `[patch."https://github.com/utensils/mold.git"]` section, then run `cargo update -p mold-ai-inference`.
