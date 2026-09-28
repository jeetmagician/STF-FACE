"""OpenCV YuNet + SFace backend.

Default backend. Both models come from the OpenCV Zoo under Apache-2.0, so
this stack can be deployed commercially without further licensing. Accuracy is
below ArcFace-R50, particularly on large age gaps and post-surgical pairs -
see docs/ARCHITECTURE.md for the trade-off.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.models.alignment import align_face, apply_affine, estimate_pose
from app.models.base import DetectedFace, FaceEmbedder, ModelInfo, l2_normalise

logger = logging.getLogger(__name__)

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"


class OpenCVEmbedder(FaceEmbedder):
    """YuNet detection + SFace 128-d recognition, both via OpenCV's DNN runtime."""

    def __init__(
        self,
        model_dir: Path,
        detection_threshold: float = 0.6,
        use_gpu: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.detection_threshold = detection_threshold

        yunet_path = self.model_dir / YUNET_FILENAME
        sface_path = self.model_dir / SFACE_FILENAME
        for path in (yunet_path, sface_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing model weights: {path}. "
                    "Run `python scripts/download_models.py` first."
                )

        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CPU
        if use_gpu:
            backend_id = cv2.dnn.DNN_BACKEND_CUDA
            target_id = cv2.dnn.DNN_TARGET_CUDA

        # input_size is reset per-image in detect(); 320x320 is a placeholder.
        self._detector = cv2.FaceDetectorYN.create(
            model=str(yunet_path),
            config="",
            input_size=(320, 320),
            score_threshold=detection_threshold,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=backend_id,
            target_id=target_id,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            model=str(sface_path),
            config="",
            backend_id=backend_id,
            target_id=target_id,
        )

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            backend="opencv",
            detector="YuNet (face_detection_yunet_2023mar)",
            recognizer="SFace (face_recognition_sface_2021dec)",
            embedding_dim=128,
            license="Apache-2.0",
            commercial_use=True,
            notes=(
                "OpenCV Zoo models. Lower accuracy than ArcFace-R50 on large age "
                "gaps and post-surgical pairs. Provides 5 facial keypoints only, "
                "so landmark-based region geometry is coarser than with the "
                "InsightFace backend."
            ),
            extras={"landmark_points": "5", "published_cosine_threshold": "0.363"},
        )

    @staticmethod
    def _order_keypoints(raw: np.ndarray) -> np.ndarray:
        """Normalise YuNet's keypoint order to [img-left eye, img-right eye,
        nose, img-left mouth, img-right mouth].

        Sorting by x rather than trusting the documented order makes this robust
        to convention changes between model releases.
        """
        points = raw.reshape(5, 2).astype(np.float32)
        eyes = points[:2][np.argsort(points[:2, 0])]
        nose = points[2:3]
        mouth = points[3:5][np.argsort(points[3:5, 0])]
        return np.vstack([eyes, nose, mouth]).astype(np.float32)

    def detect(self, image_bgr: np.ndarray, max_faces: int = 8) -> list[DetectedFace]:
        height, width = image_bgr.shape[:2]
        self._detector.setInputSize((width, height))

        try:
            _, raw_faces = self._detector.detect(image_bgr)
        except cv2.error as exc:  # pragma: no cover - defensive
            logger.warning("YuNet detection failed: %s", exc)
            return []

        if raw_faces is None or len(raw_faces) == 0:
            return []

        rows = [row for row in raw_faces if float(row[14]) >= self.detection_threshold]
        # Largest face first: the subject is almost always the dominant face.
        rows.sort(key=lambda r: float(r[2]) * float(r[3]), reverse=True)
        rows = rows[:max_faces]

        faces: list[DetectedFace] = []
        for idx, row in enumerate(rows):
            x, y, w, h = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            keypoints = self._order_keypoints(row[4:14])
            aligned, matrix = align_face(image_bgr, keypoints)
            pose = estimate_pose(keypoints, image_bgr.shape[:2])

            faces.append(
                DetectedFace(
                    index=idx,
                    bbox=(x, y, x + w, y + h),
                    detection_score=float(row[14]),
                    keypoints_5=keypoints,
                    aligned=aligned,
                    landmarks=keypoints,  # only 5 points available
                    landmarks_aligned=apply_affine(keypoints, matrix),
                    pose=pose,
                )
            )
        return faces

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (112, 112):
            aligned_bgr = cv2.resize(aligned_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        if aligned_bgr.dtype != np.uint8:
            aligned_bgr = np.clip(aligned_bgr, 0, 255).astype(np.uint8)
        feature = self._recognizer.feature(aligned_bgr)
        return l2_normalise(np.asarray(feature, dtype=np.float32).ravel())
