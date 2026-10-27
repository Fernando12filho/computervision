"""CAP 5410 - Assignment 2: image filters.

Manual (no OpenCV) implementations of the box filter and the Sobel
filters, plus the OpenCV versions used for comparison.
"""
import numpy as np
import cv2


# ---------------------------------------------------------------
# Manual implementations (no OpenCV)
# ---------------------------------------------------------------

def convolve_manual(image, kernel):
    """Slide `kernel` over a grayscale `image` and return a float64 result.

    Every output pixel is the sum of the k x k neighborhood multiplied
    element-wise by the kernel (correlation, the same convention used by
    OpenCV's filter2D/Sobel). The image is padded by reflection so that
    border pixels also have a full neighborhood.
    """
    k = kernel.shape[0]
    r = k // 2
    padded = np.pad(image.astype(np.float64), r, mode="reflect")
    out = np.zeros(image.shape, dtype=np.float64)
    rows, cols = image.shape
    for y in range(rows):
        for x in range(cols):
            window = padded[y:y + k, x:x + k]
            out[y, x] = np.sum(window * kernel)
    return out


def box_filter_manual(image, k):
    """k x k box (mean) filter: every pixel becomes the average of its neighborhood."""
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    out = convolve_manual(image, kernel)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def sobel_kernels(k):
    """Return the (Kx, Ky) Sobel kernels for k = 3 or k = 5.

    Each Sobel kernel is the outer product of a smoothing vector and a
    derivative vector. Ky is the transpose of Kx.
    """
    if k == 3:
        smooth = np.array([1, 2, 1], dtype=np.float64)
        deriv = np.array([-1, 0, 1], dtype=np.float64)
    elif k == 5:
        smooth = np.array([1, 4, 6, 4, 1], dtype=np.float64)
        deriv = np.array([-1, -2, 0, 2, 1], dtype=np.float64)
    else:
        raise ValueError("kernel size must be 3 or 5")
    kx = np.outer(smooth, deriv)
    return kx, kx.T


def sobel_x_manual(image, k):
    """Horizontal gradient Gx (responds to vertical edges). Signed float64."""
    kx, _ = sobel_kernels(k)
    return convolve_manual(image, kx)


def sobel_y_manual(image, k):
    """Vertical gradient Gy (responds to horizontal edges). Signed float64."""
    _, ky = sobel_kernels(k)
    return convolve_manual(image, ky)


def sobel_xy_manual(image, k):
    """Gradient magnitude sqrt(Gx^2 + Gy^2) from the manual Sobel filters."""
    gx = sobel_x_manual(image, k)
    gy = sobel_y_manual(image, k)
    return np.sqrt(gx ** 2 + gy ** 2)


# ---------------------------------------------------------------
# OpenCV implementations
# ---------------------------------------------------------------

def box_filter_cv(image, k):
    return cv2.blur(image, (k, k))


def sobel_xy_cv(image, k):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k)
    return cv2.magnitude(gx, gy)


def gaussian_cv(image, k):
    # sigma = 0 lets OpenCV derive sigma from the kernel size
    return cv2.GaussianBlur(image, (k, k), 0)


# ---------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------

def to_display(gradient):
    """Map a signed gradient to 0-255 for display: |g| scaled by its maximum."""
    mag = np.abs(gradient)
    return (mag / mag.max() * 255).astype(np.uint8)
