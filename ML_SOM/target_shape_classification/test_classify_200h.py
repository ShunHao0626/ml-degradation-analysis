"""Behavioral checks for strict 0–200 h selection and small temporal events."""
from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

import classify_200h
import unsupervised_som as som


class WindowClassificationTests(unittest.TestCase):
    def write_curve(self, root: Path, points):
        file = root / "curve.csv"
        with file.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "y"))
            writer.writerows(points)
        return dict(y_kind="pce", time_factor="1", folder="data_all",
                    analysis_version="curve.csv", source_csv="curve.csv")

    def test_after_200_cannot_affect_values_or_normalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.write_curve(root, [(0, 1), (50, .8), (100, .95),
                                          (190, .9), (201, 1000)])
            first, reason = classify_200h.window_observations(root, row)
            self.assertEqual(reason, "")
            row = self.write_curve(root, [(0, 1), (50, .8), (100, .95),
                                          (190, .9), (201, -1000)])
            second, reason = classify_200h.window_observations(root, row)
            self.assertEqual(reason, "")
            np.testing.assert_array_equal(first["t"], second["t"])
            np.testing.assert_array_equal(first["y"], second["y"])
            self.assertEqual(first["divisor"], 1)
            self.assertEqual(float(first["t"][-1]), 190)
            self.assertIsNotNone(classify_200h.valley_event(first["t"], first["y"]))

    def test_one_percent_fluctuations_enter_features_but_single_jitter_is_not_valley(self):
        t = np.array([0, 25, 50, 75, 100], dtype=float)
        y = np.array([1, .98, .99, .96, .95], dtype=float)
        self.assertIsNone(classify_200h.valley_event(t, y))
        from run import shape_view
        view = shape_view({"t": t, "y": y}, som.N_GRID)[0]
        _, _, _, events = som.features(np.asarray([view]), [(t / 100, y)],
                                       derivative_weight=.3)
        self.assertGreaterEqual(int(events[0]), 1)

    def test_supported_small_valley_is_kept(self):
        t = np.array([0, 20, 40, 60, 80, 100], dtype=float)
        y = np.array([1, .995, .985, .975, .98, .99], dtype=float)
        event = classify_200h.valley_event(t, y)
        self.assertIsNotNone(event)
        self.assertAlmostEqual(event["trough_h"], 60)
        self.assertLess(event["recovery"], .03)

    def test_boundary_recovery_is_flagged_without_entering_200h_window(self):
        t = np.array([0, 80, 160, 200, 220, 250], dtype=float)
        y = np.array([1, .9, .8, .78, .785, .79], dtype=float)
        self.assertIsNone(classify_200h.valley_event(t[:4], y[:4]))
        event = classify_200h.boundary_valley_event(t, y)
        self.assertIsNotNone(event)
        self.assertEqual(event["trough_h"], 200)
        self.assertAlmostEqual(event["recovery"], .01)
        self.assertIsNone(classify_200h.boundary_valley_event(
            np.array([0, 180, 220, 250.]), np.array([1, .9, .8, .9])))

    def test_short_curve_stays_short_and_unresolved_when_one_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.write_curve(root, [(0, 1), (50, .9), (500, .1)])
            window, reason = classify_200h.window_observations(root, row)
            self.assertEqual(reason, "")
            self.assertEqual(list(window["t"]), [0, 50])
            row = self.write_curve(root, [(0, 1), (500, .1)])
            window, reason = classify_200h.window_observations(root, row)
            self.assertIsNone(window)
            self.assertEqual(reason, "fewer_than_two_observations_0_200h")

    def test_200h_source_review_requires_current_image_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "source.png"
            image.write_bytes(b"current source panel")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            reviews = root / "reviews.csv"
            with reviews.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("file_id", "review_status", "reviewed_class",
                                 "source_image_sha256", "review_date", "notes"))
                writer.writerow(("F1", "verified", "bridge", digest, "2026-09-24", "rise"))
            canonical = [{"file_id": "F1", "source_image": "source.png",
                          "image_sha256": digest}]
            self.assertEqual(classify_200h.load_source_reviews(root, canonical, reviews)
                             ["F1"]["reviewed_class"], "bridge")
            image.write_bytes(b"replaced source panel")
            with self.assertRaisesRegex(ValueError, "has changed"):
                classify_200h.load_source_reviews(root, canonical, reviews)


if __name__ == "__main__":
    unittest.main()
