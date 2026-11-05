"""Independent mathematical identities and distributional checks."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from power_study_v4 import (np, basis, transform, learn, predict, assemble,
                            normalized_jacobi, gamma, matrix, f, TIMES, X, TRAIN)

class CoreChecks(unittest.TestCase):
    def setUp(self):
        self.indices = np.array([0, 1, 2, 9, 20, 31, 38, 39, 40])
        self.x = X[self.indices]

    def test_classical_finite_difference_limit(self):
        n = 31
        h = 2 / (n + 1)
        exact = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2
        np.testing.assert_allclose(matrix(n, 2.0), exact, rtol=0, atol=1e-11)

    def test_analytic_poisson_identity(self):
        alpha = 1.5
        _, _, polynomials, nodes, weights, _, stiffness = assemble(alpha, 48)
        load = polynomials.T @ (weights * np.full(len(nodes), gamma(alpha + 1)))
        coefficients = np.linalg.solve(stiffness, load)
        points = np.linspace(-.99, .99, 101)
        expected = (1 - points**2)**(alpha / 2)
        actual = expected * (normalized_jacobi(alpha, 48, points) @ coefficients)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_correlated_projection_identities(self):
        covariance = .3 * np.eye(9) + .7 * np.exp(-abs(self.x[:, None] - self.x[None, :]) / .15)
        projection, _ = transform(self.x, TRAIN[:, self.indices], C=covariance)
        np.testing.assert_allclose(projection @ basis(self.x), 0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(projection @ covariance @ projection.T,
                                   np.eye(6), rtol=0, atol=1e-12)

    def test_frozen_subspace_is_orthonormal(self):
        _, directions, _, rank, _ = learn(self.x, TRAIN[:, self.indices])
        np.testing.assert_allclose(directions[:rank] @ directions[:rank].T,
                                   np.eye(rank), rtol=0, atol=1e-12)

    def test_gaussian_null_with_independent_readouts(self):
        projection, directions, _, rank, _ = learn(self.x, TRAIN[:, self.indices])
        rng = np.random.default_rng(862026)
        errors = rng.normal(size=(2, 10000, 9, 5)) * .013
        spectral_mean = basis(self.x) @ rng.normal(size=(3, 5))
        average = spectral_mean + (errors[0] + errors[1]) / 2
        difference = (errors[0] - errors[1]) / np.sqrt(2)
        variance = np.sum(difference**2, axis=(1, 2)) / 45
        z = np.einsum('pm,rmj->rpj', projection, average).reshape(10000, -1)
        statistic = 2 * np.sum((z @ directions[:rank].T)**2, axis=1) / (rank * variance)
        rate = np.mean(statistic > f.ppf(.95, rank, 45))
        self.assertLess(abs(rate - .05), 5 * np.sqrt(.05 * .95 / 10000))

    def test_independent_basis_refinement(self):
        low = predict(1.85, .45, self.x, TIMES, 96, coeff=np.array([1, .22, .12]))
        high = predict(1.85, .45, self.x, TIMES, 128, coeff=np.array([1, .22, .12]))
        self.assertLess(float(np.max(abs(low - high))), 1e-6)

if __name__ == '__main__':
    unittest.main()
