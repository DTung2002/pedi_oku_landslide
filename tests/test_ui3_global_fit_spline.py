import csv
import io
import unittest
from unittest import mock

import numpy as np

from pedi_oku_landslide.application.ui3.exports import (
    load_theta_csv_group_angles,
    save_theta_csv_for_line,
)
from pedi_oku_landslide.domain.ui3.global_fit_spline import (
    build_global_forward_fit_spline,
    evaluate_global_fit_spline,
)


class GlobalFitSplineTests(unittest.TestCase):
    class _NonClosingStringIO(io.StringIO):
        def close(self):
            self.seek(0)

    def _profile(self):
        chain = np.linspace(0.0, 10.0, 101, dtype=float)
        elev = 20.0 - chain
        return {
            "chain": chain,
            "elev": elev,
            "elev_s": elev,
            "elev_orig": elev,
            "d_para": np.ones_like(chain),
            "dz": np.full_like(chain, -1.0),
            "theta": np.full_like(chain, -45.0),
            "slip_span": (0.0, 10.0),
        }

    def _groups(self):
        return [
            {"id": "G1", "start": 0.0, "end": 3.0},
            {"id": "G2", "start": 3.0, "end": 7.0},
            {"id": "G3", "start": 7.0, "end": 10.0},
        ]

    def test_save_theta_csv_writes_one_row_per_group(self):
        prof = self._profile()
        groups = self._groups()
        out_csv = "sandbox_theta.csv"
        buffer = self._NonClosingStringIO()
        with mock.patch("os.makedirs"), mock.patch("builtins.open", return_value=buffer):
            saved = save_theta_csv_for_line(
                line_id="ML1",
                prof=prof,
                groups=groups,
                out_csv=out_csv,
            )
        self.assertEqual(saved, out_csv)
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(len(rows), len(groups))
        self.assertEqual([row["group_id"] for row in rows], ["G1", "G2", "G3"])
        self.assertEqual([float(row["boundary_chainage"]) for row in rows], [3.0, 7.0, 10.0])

    def test_load_theta_csv_group_angles_maps_rows_to_groups(self):
        prof = self._profile()
        groups = self._groups()
        out_csv = "sandbox_theta.csv"
        write_buffer = self._NonClosingStringIO()
        with mock.patch("os.makedirs"), mock.patch("builtins.open", return_value=write_buffer):
            save_theta_csv_for_line(
                line_id="ML1",
                prof=prof,
                groups=groups,
                out_csv=out_csv,
            )
        read_buffer = self._NonClosingStringIO(write_buffer.getvalue())
        with mock.patch("os.path.exists", return_value=True), mock.patch("builtins.open", return_value=read_buffer):
            theta_rows = load_theta_csv_group_angles(csv_path=out_csv, groups=groups)
        self.assertEqual(len(theta_rows), len(groups))
        self.assertEqual([row["group_id"] for row in theta_rows], ["G1", "G2", "G3"])
        self.assertEqual([float(row["boundary_chainage"]) for row in theta_rows], [3.0, 7.0, 10.0])
        self.assertTrue(all(np.isfinite(float(row["theta_deg"])) for row in theta_rows))

    def test_global_forward_fit_spline_hits_all_fit_points(self):
        prof = self._profile()
        groups = self._groups()
        theta_rows = [
            {"group_id": "G1", "theta_deg": -30.0, "boundary_chainage": 3.0, "theta_source": "theta_percentile_20"},
            {"group_id": "G2", "theta_deg": -20.0, "boundary_chainage": 7.0, "theta_source": "median_theta_vector"},
            {"group_id": "G3", "theta_deg": -10.0, "boundary_chainage": 10.0, "theta_source": "median_theta_vector"},
        ]
        result = build_global_forward_fit_spline(prof, groups, theta_rows)

        fit_points = result.get("fit_points", [])
        self.assertEqual(len(fit_points), 7)
        self.assertEqual(len(result.get("global_fit_points", [])), 7)
        self.assertEqual(len(result.get("steps", [])), 3)
        self.assertEqual(len(result.get("boundary_intersections", [])), 2)

        fit_chains = [float(pt["chain"]) for pt in fit_points]
        self.assertAlmostEqual(fit_chains[0], 0.0, places=6)
        self.assertAlmostEqual(fit_chains[2], 3.0, places=5)
        self.assertAlmostEqual(fit_chains[4], 7.0, places=5)
        self.assertAlmostEqual(fit_chains[-1], 10.0, places=6)

        spline_def = result["spline_definition"]
        fit_params = np.asarray(spline_def.get("fit_parameters", []), dtype=float)
        self.assertEqual(fit_params.size, len(fit_points))
        for fit_param, fit_point in zip(fit_params, fit_points):
            x_val, z_val = evaluate_global_fit_spline(spline_def, float(fit_param))
            self.assertAlmostEqual(float(x_val), float(fit_point["chain"]), places=5)
            self.assertAlmostEqual(float(z_val), float(fit_point["elev"]), places=5)

        for hit in result.get("boundary_intersections", []):
            self.assertAlmostEqual(float(hit["chain"]), float(hit["boundary_x"]), places=5)


if __name__ == "__main__":
    unittest.main()
