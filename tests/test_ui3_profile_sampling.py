import unittest

import numpy as np

from pedi_oku_landslide.application.ui3.profile_sampling import (
    fixed_chainage_grid,
    parse_nominal_length_m,
    resample_profile_to_nominal_grid,
)


class ProfileSamplingTests(unittest.TestCase):
    def test_parse_nominal_length(self):
        self.assertEqual(parse_nominal_length_m("ML1__(130.0_m)"), 130.0)
        self.assertIsNone(parse_nominal_length_m("ML1"))

    def test_fixed_chainage_grid(self):
        got = fixed_chainage_grid(1.0, 0.2)
        np.testing.assert_allclose(got, np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float))

    def test_resample_profile_to_nominal_grid(self):
        prof = {
            "chain": np.asarray([0.0, 0.5, 1.0], dtype=float),
            "elev": np.asarray([10.0, 9.0, 8.0], dtype=float),
            "elev_s": np.asarray([10.0, 9.0, 8.0], dtype=float),
            "dz": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "slip_mask": np.asarray([False, True, True]),
            "slip_span": (0.5, 1.0),
        }
        out = resample_profile_to_nominal_grid(
            prof,
            line_id="ML1__(1.0_m)",
            target_step_m=0.2,
            nominal_length_m=1.0,
        )
        np.testing.assert_allclose(
            np.asarray(out["chain"], dtype=float),
            np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(out["elev"], dtype=float),
            np.asarray([10.0, 9.6, 9.2, 8.8, 8.4, 8.0], dtype=float),
            atol=1e-6,
        )
        self.assertEqual(tuple(map(bool, out["slip_mask"].tolist())), (False, False, True, True, True, True))


if __name__ == "__main__":
    unittest.main()
