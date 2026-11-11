"""Unit tests for PRISM perception geometry and runtime helpers."""

import unittest

from perception.detection import (
    compute_approach_velocity,
    compute_centroid,
    compute_velocity,
    resolve_device,
    resolve_weights_path,
    smooth_height,
)


class DetectionHelpersTest(unittest.TestCase):
    def test_resolve_device_selects_cuda_or_cpu(self) -> None:
        self.assertEqual(resolve_device(True), "cuda")
        self.assertEqual(resolve_device(False), "cpu")

    def test_default_weights_path_points_to_existing_yolov8n(self) -> None:
        weights = resolve_weights_path()
        self.assertEqual(weights.name, "yolov8n.pt")
        self.assertTrue(weights.is_file())

    def test_centroid_and_velocity_calculations(self) -> None:
        cx, cy = compute_centroid(10, 20, 50, 100)
        self.assertEqual((cx, cy), (30, 60))

        self.assertEqual(compute_velocity(cx, cy, 1.0, None), (0.0, 0.0, 0.0))
        vx, vy, speed = compute_velocity(40, 80, 3.0, (10, 40, 1.0))
        self.assertAlmostEqual(vx, 15.0)
        self.assertAlmostEqual(vy, 20.0)
        self.assertAlmostEqual(speed, 25.0)

    def test_height_smoothing_and_approach_velocity(self) -> None:
        self.assertEqual(smooth_height(100.0, None), 100.0)
        smoothed = smooth_height(200.0, 100.0, alpha=0.7)
        self.assertAlmostEqual(smoothed, 130.0)

        self.assertEqual(compute_approach_velocity(smoothed, 2.0, None), 0.0)
        v_approach = compute_approach_velocity(130.0, 3.0, (100.0, 1.0))
        self.assertAlmostEqual(v_approach, 15.0)

        with self.assertRaises(ValueError):
            smooth_height(100.0, 90.0, alpha=1.5)


if __name__ == "__main__":
    unittest.main()
