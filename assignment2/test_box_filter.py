import numpy as np
import cv2
from filters import box_filter_manual, sobel_x_manual, sobel_y_manual


def test_output_has_same_shape_as_input():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (20, 30), dtype=np.uint8)
    for k in (3, 5):
        assert box_filter_manual(img, k).shape == img.shape


def test_constant_image_stays_constant():
    # Averaging identical values must give the same value back.
    # Catches wrong divisors AND uint8 overflow while summing.
    img = np.full((10, 10), 100, dtype=np.uint8)
    for k in (3, 5):
        out = box_filter_manual(img, k)
        r = k // 2
        assert np.allclose(out[r:-r, r:-r], 100)


def test_vertical_edge_becomes_ramp():
    # The worked example from our chat: 0 | 255 edge -> 85, 170
    img = np.array([[0, 0, 255, 255]] * 3, dtype=np.uint8)
    out = box_filter_manual(img, 3)
    assert np.isclose(out[1, 1], 85)
    assert np.isclose(out[1, 2], 170)


def test_matches_opencv_in_interior():
    # cv2 is used ONLY here, as a reference to check against.
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (50, 60), dtype=np.uint8)
    for k in (3, 5):
        mine = box_filter_manual(img, k).astype(float)
        ref = cv2.blur(img, (k, k)).astype(float)
        r = k // 2
        assert np.abs(mine[r:-r, r:-r] - ref[r:-r, r:-r]).max() <= 1


def test_sobel_matches_opencv():
    # Same border handling (reflect) as OpenCV's default, so every pixel matches.
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    for k in (3, 5):
        assert np.allclose(sobel_x_manual(img, k), cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k))
        assert np.allclose(sobel_y_manual(img, k), cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k))
