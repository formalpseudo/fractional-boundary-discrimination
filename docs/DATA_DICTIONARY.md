# Synthetic data dictionary

All paths below are relative to `code/outputs`.

| File | Contents and interpretation |
|:--|:--|
| `v3/design_means.npz` | `candidate`: 41 locations; `train`: 24 × 41 × 5 means; `holdout`: 12 × 41 × 5 means |
| `v3/design_training_scenarios.csv` | Training order, diffusivity, second and third initial-mode coefficients |
| `v3/sensor_designs.json` | Energy-based and control sensor sets at budgets 7, 9, 13 and 17 |
| `v3/sensor_designs_2d.json` | Frozen 2D locations for uniform, central and robust designs |
| `v3/aperture*.csv` | Signal energies and fitted sixth/eighth-order slopes |
| `v4/frozen_designs.json` | Geometry indices/positions, selected rank and training score |
| `v4/test_scenarios.csv` | 524 alternatives: 256 interpolation, 256 stress, 12 benchmark cases labelled `legacy` |
| `v4/test_means.npz` | `means`: 524 × 41 × 5 restricted-model observations |
| `v4/holdout.csv` | 100,608 rows: four budgets × three geometry labels × four methods × four noise levels × 524 scenarios |
| `v4/rank_ablation.csv` | 11,004 rows: seven ranks × three geometry labels × 524 scenarios |
| `v4/calibration.csv` | Rejection counts summarized as proportions and pointwise Wilson intervals; 5,000 repeats per configuration |
| `v4/paired_residual.csv` | 1,800 fitted datasets, optimizer status, fitted parameters and three paired diagnostic decisions |
| `v4/noise_stress.csv` | Gaussian/Student-t4 calibration and power under four known covariance shapes |
| `v4/support_stress.csv` | Three declared supports, four fifth-mode amplitudes, paired diagnostic results |
| `v4/twod_holdout.csv` | 2,048 exact-power evaluations over 128 unseen directional-order pairs |
| `v4/reference_refinement.csv` | 96-to-128-function discrepancies on selected held-out means |
| `v4/*_runtime.json` | Execution time, source hash and seed policy for the original run |

`noise` is a fraction of the fixed amplitude reference 1.1615502616178095; 0.01 means 1%. `power`, `rate`, `low` and `high` are proportions. `retained` is compressed divided by full projected energy on the same geometry. `m` is sensor count; `rank` is numerator degrees of freedom. `case` identifies the corresponding row of the scenario file. `alpha`, `kappa`, `c2`, and `c3` define each one-dimensional alternative; the first initial amplitude is one. Space, time and coefficients are dimensionless in these benchmarks.

The `energy` and `power` nine-sensor layouts coincide. Their duplicate rows must be treated as paired results. Training/testing use row-major sensor-time vectorization consistently. Seeds are explicit constants in the source. The readout noise used for calibration is independent of the simulations used to select geometry and rank.
