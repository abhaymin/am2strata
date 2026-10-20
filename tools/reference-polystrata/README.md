# Reference recorders (run the ORIGINAL PolyStrata)

These scripts import PolyStrata's own Python code (`~/Development/PolyStrata`) and record what it does - prompts,
token ids, parser events, tokenizer ids - as the golden files in `golden/` and `crates/*/testdata/`. am2Strata's Rust
tests compare against those files. They are not part of am2Strata: nothing runs them at build, test or serve time,
and they stay Python only because the thing they record is Python.

    ~/Development/PolyStrata/.venv/bin/python tools/reference-polystrata/make_golden.py
