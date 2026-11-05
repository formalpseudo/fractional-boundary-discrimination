# Release validation

The six numerical unit tests passed in Python 3.12.14 with the versions listed in `requirements-tested.txt`. The saved-result validator passed all 29 checks.

The executable example freshly recomputed all 24 training means, the twelve benchmark alternatives, and 256 stress alternatives. The maximum absolute difference in recomputed powers was 3.22e-15. The selected nine-sensor compressed rank is two. See `reports/CORE_REPRODUCTION.json` and `reports/VALIDATION_V4.json` for actual values.

The manuscript description of this selected rank is corrected to two in the code-availability revision. The previously reported detection probabilities already used the correct rank and remain unchanged. The two-dimensional experiment independently selects rank one for both layouts.

Continuous integration repeats the unit tests, fresh core reproduction, and saved-result validation on Linux. The workflow status on GitHub records that independent run.
