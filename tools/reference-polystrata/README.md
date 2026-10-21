# Reference recorders (run the ORIGINAL PolyStrata)

These scripts import PolyStrata's own Python code (`~/Development/PolyStrata`) and record what it does - prompts,
token ids, parser events, tokenizer ids - as the golden files in `golden/` and `crates/*/testdata/`. am2Strata's Rust
tests compare against those files. They are not part of am2Strata: nothing runs them at build, test or serve time,
and they stay Python only because the thing they record is Python.

    ~/Development/PolyStrata/.venv/bin/python tools/reference-polystrata/make_golden.py
    ~/Development/PolyStrata/.venv/bin/python tools/reference-polystrata/make_golden_pack.py

`make_golden_pack.py` records PolyStrata's packing code for crate `strata-pack` (`golden/pack/`): gguf-py's
dequantizers on random blocks, numpy's f16/BF16 rounding and Python's float repr, mtp_pack.py / mtp_rt.py's
quantizers, a tiny two-shard model packed by tools/iq_pack.py, and a tiny MTP source packed by tools/mtp_pack.py.
