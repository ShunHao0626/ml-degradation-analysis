"""Checks for the scientific failure modes in the new early-window pipeline."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_data import features, prepare
from diagram_stage_classifier import classify
from masked_som import MaskedSOM


CONFIG = {"window_h": 200, "grid_step_h": 2, "maximum_interpolation_gap_h": 40}


class EarlyWindowTests(unittest.TestCase):
    def test_diagram_stages_and_real_turning_points(self):
        t = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 300], dtype=float)
        examples = {
            "bridge": [.5, .7, .9, 1, .99, .98, .97, .96, .95, .94, .93, .91, .89],
            "hill": [.5, .7, 1, .90, .80, .72, .68, .65, .63, .61, .60, .59, .58],
            "slope": [1, .8, .65, .55, .50, .47, .45, .43, .42, .41, .40, .39, .38],
            "valley": [1, .8, .6, .5, .55, .65, .75, .8, .85, .87, .88, .86, .84],
        }
        for expected, values in examples.items():
            with self.subTest(expected=expected):
                result = classify(t, np.array(values))
                self.assertEqual(result["candidate_class"], expected)
                self.assertEqual(result["evidence_status"], "stage_candidate")
                self.assertIn(result["primary_turn_h"], set(t))
                if result["secondary_turn_h"] != "":
                    self.assertIn(result["secondary_turn_h"], set(t))

    def test_post_200_recovery_does_not_become_strict_valley(self):
        t = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240], dtype=float)
        y = np.array([1, .9, .8, .7, .6, .5, .4, .3, .2, .1, .05, .2, .3])
        result = classify(t, y)
        self.assertEqual(result["candidate_class"], "slope")
        self.assertTrue(result["boundary_valley_hint"])

    def test_post_200_change_cannot_change_strict_shape(self):
        t = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300], dtype=float)
        early = [.5, .7, .9, 1, .99, .98, .97, .96, .95, .94, .93]
        falling = classify(t, np.array(early + [.8, .6]))
        rising = classify(t, np.array(early + [1.2, 1.5]))
        self.assertEqual(falling["candidate_class"], rising["candidate_class"])
        self.assertEqual(falling["primary_turn_h"], rising["primary_turn_h"])

    def test_window_denominator_duplicate_times_and_boundary_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "curve.csv"
            source.write_text("x,y\n0,8\n10,10\n10,9\n20,9\n220,100\n")
            row = {"source_csv": "curve.csv", "folder": "data_all",
                   "time_factor": "1", "y_kind": "pce"}
            config = dict(CONFIG, boundary_limit_h=250, boundary_max_points=3)
            item, error = prepare(row, directory, config)
            self.assertEqual(error, "")
            self.assertEqual(item["divisor"], 10)
            self.assertEqual(item["duplicates"], 1)
            self.assertAlmostEqual(item["y"][1], .95)
            original = features(item, config)
            source.write_text("x,y\n0,8\n10,10\n10,9\n20,9\n220,1000\n")
            changed, error = prepare(row, directory, config)
            self.assertEqual(error, "")
            for a, b in zip(original, features(changed, config)):
                np.testing.assert_array_equal(a, b)

    def test_late_values_cannot_change_strict_features(self):
        item = {"t": np.array([0., 10., 20., 40.]),
                "y": np.array([.8, 1., .98, .96]),
                "boundary": [(220, .2, 5)], "points": [(0, .8, 2), (220, .2, 5)]}
        before = features(item, CONFIG)
        changed = copy.deepcopy(item)
        changed["boundary"] = [(220, 20., 5)]
        changed["points"][-1] = (220, 20., 5)
        for a, b in zip(before, features(changed, CONFIG)):
            np.testing.assert_array_equal(a, b)

    def test_short_record_and_large_gap_are_masked(self):
        short = {"t": np.array([0., 20.]), "y": np.array([1., .9])}
        trajectory = features(short, CONFIG)[0]
        self.assertTrue(np.isnan(trajectory[11:]).all())
        gap = {"t": np.array([0., 80., 100.]), "y": np.array([1., .9, .8])}
        trajectory = features(gap, CONFIG)[0]
        self.assertTrue(np.isnan(trajectory[1:40]).all())
        self.assertEqual(trajectory[40], .9)

    def test_masked_som_ignores_unsupported_values(self):
        model = MaskedSOM(2, 2, [2], [1], 41)
        model.prototypes = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        model.supported = np.array([True, True])
        a = model.distance_one(np.array([.1, 999.]), np.array([True, False]))
        b = model.distance_one(np.array([.1, -999.]), np.array([True, False]))
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
