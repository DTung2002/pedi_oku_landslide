import math
import unittest

import numpy as np

from pedi_oku_landslide.application.ui3.workflows import evaluate_nurbs
from pedi_oku_landslide.domain.ui3.curve_state import build_default_nurbs_params


class SegmentedSlopeGuidedTests(unittest.TestCase):
    def _profile(self):
        chain = np.arange(0.0, 10.0 + 1.0, 1.0, dtype=float)
        elev = 10.0 - chain
        d_para = np.ones_like(chain)
        dz = np.full_like(chain, -0.25)
        dz[(chain >= 3.0) & (chain <= 7.0)] = -1.0
        dz[(chain >= 7.0) & (chain <= 10.0)] = -0.5
        return {
            "chain": chain,
            "elev_s": elev,
            "elev": elev,
            "elev_orig": elev,
            "d_para": d_para,
            "dz": dz,
        }

    def _groups(self):
        return [
            {"id": "G1", "start": 0.0, "end": 3.0},
            {"id": "G2", "start": 3.0, "end": 7.0},
            {"id": "G3", "start": 7.0, "end": 10.0},
        ]

    def test_segmented_builder_uses_group_boundaries_and_slopes(self):
        prof = self._profile()
        groups = self._groups()
        params = build_default_nurbs_params(
            prof=prof,
            groups=groups,
            base_curve={},
            endpoints=(0.0, 10.0, 10.0, 0.0),
            nurbs_seed_method="slope_guided",
        )
        cps = np.asarray(params.get("control_points", []), dtype=float)
        self.assertEqual(cps.shape, (4, 2))
        np.testing.assert_allclose(cps[:, 0], np.asarray([0.0, 3.0, 7.0, 10.0], dtype=float))
        np.testing.assert_allclose(cps[:, 1], np.asarray([10.0, 7.0, 3.0, 0.0], dtype=float), atol=1e-6)

        segments = params.get("segments", [])
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["theta_source"], "theta_percentile_20")
        self.assertEqual(segments[1]["theta_source"], "median_theta_vector")
        self.assertEqual(segments[2]["theta_source"], "median_theta_vector")
        self.assertAlmostEqual(float(segments[0]["theta_deg"]), -45.0, places=6)
        self.assertAlmostEqual(float(segments[1]["theta_deg"]), -45.0, places=6)
        self.assertAlmostEqual(float(segments[2]["theta_deg"]), math.degrees(math.atan(-0.5)), places=6)

    def test_piecewise_evaluator_hits_intersections_once(self):
        prof = self._profile()
        groups = self._groups()
        params = build_default_nurbs_params(
            prof=prof,
            groups=groups,
            base_curve={},
            endpoints=(0.0, 10.0, 10.0, 0.0),
            nurbs_seed_method="slope_guided",
        )
        curve = evaluate_nurbs(params, n_samples=180)
        sx = np.asarray(curve.get("chain", []), dtype=float)
        sz = np.asarray(curve.get("elev", []), dtype=float)
        self.assertGreaterEqual(sx.size, 30)
        self.assertEqual(int(np.count_nonzero(np.isclose(sx, 3.0, atol=1e-9))), 1)
        self.assertEqual(int(np.count_nonzero(np.isclose(sx, 7.0, atol=1e-9))), 1)
        self.assertAlmostEqual(float(sx[0]), 0.0, places=6)
        self.assertAlmostEqual(float(sz[0]), 10.0, places=6)
        self.assertAlmostEqual(float(sx[-1]), 10.0, places=6)
        self.assertAlmostEqual(float(sz[-1]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
