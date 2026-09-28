"""Geometric alignment and pose estimation shared by all backends.

Implements the Umeyama similarity transform directly rather than depending on
scikit-image, keeping the install lean.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.base import ARCFACE_112_TEMPLATE

# A coarse generic 3D face model in an arbitrary metric space, ordered to match
# the 5-point keypoint convention. Used only for solvePnP pose estimation, which
# needs relative geometry rather than true anthropometric accuracy.
_GENERIC_FACE_3D = np.array(
    [
        [-30.0, 35.0, -30.0],   # left eye
        [30.0, 35.0, -30.0],    # right eye
        [0.0, 0.0, 0.0],        # nose tip
        [-25.0, -35.0, -25.0],  # left mouth corner
        [25.0, -35.0, -25.0],   # right mouth corner
    ],
    dtype=np.float64,
)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping `src` onto `dst`.

    Returns a 2x3 affine matrix suitable for cv2.warpAffine. Implements
    Umeyama (1991), "Least-squares estimation of transformation parameters
    between two point patterns".
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    covariance = dst_demean.T @ src_demean / num

    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        d[dim - 1] = -1

    transform = np.eye(dim + 1, dtype=np.float64)
    u, s, vt = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)

    if rank == 0:
        return np.eye(3, dtype=np.float64)[:2]
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            transform[:dim, :dim] = u @ vt
        else:
            s_last = d[dim - 1]
            d[dim - 1] = -1
            transform[:dim, :dim] = u @ np.diag(d) @ vt
            d[dim - 1] = s_last
    else:
        transform[:dim, :dim] = u @ np.diag(d) @ vt

    src_var = src_demean.var(axis=0).sum()
    if src_var < 1e-12:
        scale = 1.0
    else:
        scale = float((s * d).sum() / src_var)

    transform[:dim, dim] = dst_mean - scale * (transform[:dim, :dim] @ src_mean)
    transform[:dim, :dim] *= scale
    return transform[:2].astype(np.float64)


def align_face(
    image_bgr: np.ndarray, keypoints_5: np.ndarray, size: int = 112
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a face to the canonical 112x112 template.

    Returns (aligned_image, 2x3_affine_matrix).
    """
    template = ARCFACE_112_TEMPLATE.astype(np.float64)
    if size != 112:
        template = template * (size / 112.0)
    matrix = umeyama_similarity(np.asarray(keypoints_5, dtype=np.float64), template)
    aligned = cv2.warpAffine(
        image_bgr,
        matrix.astype(np.float32),
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, matrix


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map points through a 2x3 affine matrix."""
    pts = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (homogeneous @ matrix.T).astype(np.float32)


def estimate_pose(
    keypoints_5: np.ndarray, image_shape: tuple[int, int]
) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) in degrees from 5 keypoints via solvePnP.

    This is a coarse estimate. It is used for quality gating and for reporting
    pose difference, not as an identity signal.
    """
    height, width = image_shape[:2]
    focal_length = float(width)
    camera_matrix = np.array(
        [[focal_length, 0, width / 2.0], [0, focal_length, height / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points = np.asarray(keypoints_5, dtype=np.float64).reshape(-1, 2)
    try:
        ok, rotation_vector, _ = cv2.solvePnP(
            _GENERIC_FACE_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return (0.0, 0.0, 0.0)
    if not ok:
        return (0.0, 0.0, 0.0)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6
    if singular:
        pitch = float(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        yaw = float(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = 0.0
    else:
        pitch = float(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw = float(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

    yaw_deg = float(np.degrees(yaw))
    pitch_deg = float(np.degrees(pitch))
    roll_deg = float(np.degrees(roll))

    # solvePnP returns pitch near +/-180 for upright faces; fold to a sane range.
    if pitch_deg > 90:
        pitch_deg -= 180
    elif pitch_deg < -90:
        pitch_deg += 180
    if roll_deg > 90:
        roll_deg -= 180
    elif roll_deg < -90:
        roll_deg += 180

    return (yaw_deg, pitch_deg, roll_deg)


def roll_from_eyes(keypoints_5: np.ndarray) -> float:
    """Roll angle in degrees from the eye-line, as a robust cross-check."""
    left_eye, right_eye = keypoints_5[0], keypoints_5[1]
    dy = float(right_eye[1] - left_eye[1])
    dx = float(right_eye[0] - left_eye[0])
    return float(np.degrees(np.arctan2(dy, dx)))
