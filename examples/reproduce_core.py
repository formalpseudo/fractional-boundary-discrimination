"""Recompute training and held-out means, then verify key fixed-design powers."""
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from power_study_v4 import (np, pd, predict, basis, learn, transform, power,
                            TIMES, X, TRAIN, V3, V4)

def main():
    start = time.perf_counter()
    parameters = pd.read_csv(V3 / 'design_training_scenarios.csv')
    training = np.array([
        predict(p.alpha, p.kappa, X, TIMES, 96,
                coeff=np.array([1, p.c2, p.c3]))
        for p in parameters.itertuples()
    ])
    training_error = float(np.max(np.abs(training - TRAIN)))
    assert training_error < 1e-10, training_error
    designs = json.loads((V4 / 'frozen_designs.json').read_text())
    selected = designs['9:power']['indices']
    x = X[selected]
    projection, vectors, _, rank, _ = learn(x, training[:, selected])
    assert rank == designs['9:power']['rank']
    scenarios = pd.read_csv(V4 / 'test_scenarios.csv')
    saved = pd.read_csv(V4 / 'holdout.csv')
    _, nominal = transform(x, predict(1.5, .5, x, TIMES, 128)[None])
    local = nominal[0] / np.linalg.norm(nominal[0])
    summaries = {}
    largest_power_difference = 0.0
    for domain in ['legacy', 'stress']:
        subset = scenarios[scenarios.domain == domain]
        means = np.array([
            predict(p.alpha, p.kappa, x, TIMES, 128,
                    coeff=np.array([1, p.c2, p.c3]))
            for p in subset.itertuples()
        ])
        _, residuals = transform(x, means)
        alternatives = {
            'omnibus': (residuals.shape[1], np.sum(residuals**2, axis=1)),
            'compressed': (rank, np.sum((residuals @ vectors[:rank].T)**2, axis=1)),
            'local-rank1': (1, (residuals @ local)**2),
        }
        summaries[domain] = {}
        for method, (df, energy) in alternatives.items():
            values = power(energy, df, len(x) * len(TIMES), .01)
            reference = saved[(saved.m == 9) & (saved.design == 'power') &
                              (saved.noise == .01) & (saved.domain == domain) &
                              (saved.method == method)].sort_values('case').power.to_numpy()
            delta = float(np.max(np.abs(values - reference)))
            largest_power_difference = max(delta, largest_power_difference)
            assert delta < 1e-6, (domain, method, delta)
            summaries[domain][method] = {'mean': float(np.mean(values)),
                                         'minimum': float(np.min(values)),
                                         'scenarios': len(values)}
    record = {'passed': True, 'training_max_abs_difference': training_error,
              'power_max_abs_difference': largest_power_difference,
              'selected_rank': rank, 'sensor_positions': x.tolist(),
              'summaries': summaries, 'seconds': time.perf_counter() - start}
    target = ROOT / 'outputs'
    target.mkdir(exist_ok=True)
    (target / 'core_reproduction.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return record

if __name__ == '__main__':
    main()
