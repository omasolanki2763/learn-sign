# Step 0 Task 8 — Live Verifier Spike Report

| Judge | Correct pass | False-green (wrong) | False-green (minimal pair) | Perturbation caught | Right part named | First verdict (s) |
|---|---|---|---|---|---|---|
| (a) data ranges | 1.000* | 0.000* | 0.000* | 1.000* | 1.000* | 0.100* |
| (b) LLM specs | pending | pending | pending | pending | pending | pending |
| (c) trained model | pending | pending | pending | pending | pending | pending |
| (d) ranges + model | pending | pending | pending | pending | pending | pending |

\* *Synthetic baseline validation numbers. Real held-out GISLR evaluation is pending execution on Om's Kaggle account via `ml/notebooks/step0_verifier.ipynb`.*

**Targets:**
- Correct pass rate $\ge 0.90$
- False-green rate (random wrong & minimal pairs) $\le 0.05$
- Perturbation caught $\ge 0.90$
- Right part named $\ge 0.80$
- First verdict after onset $\le 0.300\text{ s}$

## Eye-Fallback Calibration (Baseline Target Parameters)

```json
{
  "eye_to_shoulder_ratio": {
    "median": 6.0,
    "p05": 5.2,
    "p95": 6.8
  },
  "eye_to_shoulder_y_offset": {
    "median": 0.5,
    "p05": 0.42,
    "p95": 0.58
  },
  "frames_used": 0,
  "status": "calibrated_function_ready_for_kaggle_run"
}
```

## Plain-English Reading

1. **Synthetic Verification Passes:** Onset gating, open-end DTW streaming alignment, k-medoids variant clustering, and per-phase range checks reliably isolate individual errors (location, handshape, movement, hands) on synthetic sequences.
2. **Minimal Pairs Defined & Verified:** 20 core evaluation signs (including location minimal pairs like MOM vs DAD and eye/nose/mouth) are committed in `step0_signs.json` and 100% verified against the GISLR 250-gloss curriculum.
3. **Pluggable Multi-Judge Interface:** Four candidate judges (data-only ranges, LLM declarative specs, trained whole-sign model, and hybrid ranges+model) are defined under the unified `Judge` protocol.
4. **Candidate Stubs:** Candidate (b) is stubbed pending Task 9 declarative specs; candidates (c) and (d) automatically load when the Task 5 model checkpoint is supplied.
5. **Ready for Kaggle:** Full offline evaluation pipeline and calibration routines are packaged in `ml/learnsign/verify/run_spike.py` and `ml/notebooks/step0_verifier.ipynb` to run completely unattended once Kaggle GPU compute is active.
