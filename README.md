# Fractional boundary discrimination

Core numerical code and synthetic benchmark records for **Frozen discrepancy subspaces and sensor design for distinguishing fractional diffusion boundary models**, by Yiheng Huang and Muzhou Hou. This repository accompanies a research manuscript prepared for *Fractal and Fractional*; it is a software release, not a claim of journal acceptance.

The code compares the spectral Dirichlet and zero-exterior integral fractional diffusion models on a bounded interval. A sampled modal projection removes the spectral nuisance mean; independent simulations define a frozen discrepancy subspace, and paired observations supply an independent variance estimate. Sensor geometry and subspace rank are selected using training alternatives.

## Quick reproduction

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/reproduce_core.py
python code/validate_v4.py
```

On Windows, activate with `.venv\Scripts\activate`. `requirements.txt` gives supported dependency ranges. `requirements-tested.txt` records the exact numerical package versions used in the original macOS/arm64 environment; cross-platform installations may use different BLAS implementations. Numerical checks therefore use tolerances.

The example recomputes all 24 training means, the twelve-case benchmark, and the 256 stress alternatives from the Galerkin solver. It checks the saved training cache, frozen rank, and published power summaries. Its JSON result is written to `outputs/core_reproduction.json`.

The frozen nine-sensor design selects rank two; the local comparator has rank one. Expected values at 1% noise are approximately:

| Evaluation | Full residual | Frozen compression | Local rank one |
|:--|--:|--:|--:|
| Worst power, twelve-case benchmark | 64.29% | 99.31% | 99.65% |
| Mean power, 256 stress cases | 53.79% | 80.11% | 80.21% |

The local comparator remains slightly stronger on these summaries. Mean stress power and worst benchmark power describe different evaluation sets.

## Implementation and manuscript mapping

| Source | Role | Manuscript |
|:--|:--|:--|
| `code/independent_galerkin.py` | Weighted-Jacobi Galerkin solver for the restricted integral operator | Section 6.1 |
| `code/run_experiments.py` | Independent centered differences, spectral solution and inverse fitting | Sections 6.2 and 7.7 |
| `code/extended_study_v3.py` | Aperture law, exchange search, variable projection and directional 2D model | Sections 3–6 |
| `code/power_study_v4.py` | Whitening, frozen SVD compression, rank/geometry selection and stress experiments | Sections 3.3, 5.3–5.4, 6.5 and 7.8–7.12 |
| `code/validate_v4.py` | Saved-result, calibration, projection and provenance checks | Computational verification |
| `code/figures_v4.py` | Seven quantitative compression figures | Figures 8–14 |

The `v3` and `v4` filenames identify the experiment implementations. They are retained unchanged to preserve the hashes in the execution records. The public software release is **v1.0.0**.

## Regenerate experiments

The distributed training means and designs allow the commands below to run directly. To regenerate those inputs first, run the two baseline design commands:

```sh
python code/extended_study_v3.py design
python code/extended_study_v3.py twod
python code/extended_study_v3.py aperture
python code/power_study_v4.py design
python code/power_study_v4.py holdout
python code/power_study_v4.py calibration
python code/power_study_v4.py robustness
python code/power_study_v4.py twod4
python code/validate_v4.py
python code/figures_v4.py
```

Run commands in this order when rebuilding everything. The numerical base module limits BLAS to one thread, matching the study protocol. Running experiment commands overwrites their corresponding generated records; retain the tagged checkout for comparison. `validate_v4.py` writes a JSON report; its `passed` field is the aggregate validation result. The unit-test command exits nonzero on failure.

## Data and scope

All observations are synthetic. The release contains the core implementations, candidate/training means, frozen sensor designs, parameter tuples, random-seed rules, exact power evaluations, Monte Carlo summaries, paired nonlinear-fit records, and the seven-figure plotting script. See [the data dictionary](docs/DATA_DICTIONARY.md).

There are 100,608 analytical power records and 11,004 rank-ablation records. They are factorial evaluations, not independent experimental replicates. Coincident sensor layouts share outcomes, and several Monte Carlo comparisons share noise draws. The conditional F law assumes declared initial support, Gaussian independent readouts and time errors, and known covariance shape. Student innovations are empirical robustness tests.

This core-code release supports the method and compression experiments. The full manuscript supplement additionally contains the earlier inverse-study records, extended bootstrap/correlation comparisons, and conceptual figures. Those manuscript and illustration files are outside this code release.

## Numerical sources

The Galerkin implementation uses the weighted-Jacobi identity described by Acosta, Borthagaray, Bruno and Maas, *Mathematics of Computation* 87 (2018), 1821–1857, https://doi.org/10.1090/mcom/3276. Centered differences follow Ortigueira, *Journal of Vibration and Control* 14 (2008), 1255–1266, https://doi.org/10.1177/1077546307087453. Variable projection follows Golub and Pereyra, *SIAM Journal on Numerical Analysis* 10 (1973), 413–432, https://doi.org/10.1137/0710036. Matched-subspace detection is described by Scharf and Friedlander, *IEEE Transactions on Signal Processing* 42 (1994), 2146–2157, https://doi.org/10.1109/78.301849. The algorithms here are project implementations; no third-party source files are vendored.

## Citation and license

Use `CITATION.cff` and the versioned [v1.0.0 release](https://github.com/formalpseudo/fractional-boundary-discrimination/releases/tag/v1.0.0). The release includes a source archive and SHA-256 checksums. The code and accompanying synthetic records are distributed under the MIT license in `LICENSE`.

AI assistance: OpenAI Codex assisted with programming, numerical execution, documentation and manuscript preparation. Scientific use should retain the stated model assumptions and benchmark definitions.
