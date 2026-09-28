"""InsightFace (SCRFD + ArcFace) backend.

LICENSING - READ BEFORE DEPLOYING
---------------------------------
The InsightFace *library* is MIT licensed. The pretrained model packs it
downloads (buffalo_l, buffalo_s, antelopev2, ...) are released by the upstream
maintainers for NON-COMMERCIAL RESEARCH USE ONLY. That applies to both manual
downloads and the library's automatic download.

Commercial deployment requires a licence from InsightFace
(recognition-oss-pack@insightface.ai), or substituting your own weights.

This backend is therefore opt-in: set MODEL_BACKEND=insightface explicitly.
The default `opencv` backend is Apache-2.0 and carries no such restriction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.models.alignment import align_face, apply_affine, estimate_pose
from app.models.base import DetectedFace, FaceEmbedder, ModelInfo, l2_normalise

logger = logging.getLogger(__name__)


class InsightFaceEmbedder(FaceEmbedder):
    """SCRFD detection + 106-point landmarks + ArcFace 512-d recognition."""

    def __init__(
        self,
        model_dir: Path,
        pack: str = "buffalo_l",
        det_size: int = 640,
        detection_threshold: float = 0.6,
        use_gpu: bool = False,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The insightface package is not installed. Install the optional "
                "extra with `pip install -r requirements-insightface.txt`, or use "
                "MODEL_BACKEND=opencv."
            ) from exc

        self.pack = pack
        self.detection_threshold = detection_threshold

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(
            name=pack,
            root=str(Path(model_dir).parent),
            providers=providers,
            allowed_modules=["detection", "landmark_2d_106", "recognition"],
        )
        self._app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))

        self._recognition_model = self._app.models.get("recognition")
        if self._recognition_model is None:  # pragma: no cover - defensive
            raise RuntimeError(
                f"The '{pack}' pack did not provide a recognition model."
            )

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            backend="insightface",
            detector=f"SCRFD ({self.pack})",
            recognizer=f"ArcFace w600k_r50 ({self.pack})",
            embedding_dim=512,
            license="Library MIT; pretrained weights non-commercial research only",
            commercial_use=False,
            notes=(
                "Higher accuracy than the OpenCV backend, especially across large "
                "age gaps and post-surgical pairs. Pretrained weights require a "
                "separate commercial licence from InsightFace before any "
                "commercial deployment."
            ),
            extras={"landmark_points": "106", "pack": self.pack},
        )

    def detect(self, image_bgr: np.ndarray, max_faces: int = 8) -> list[DetectedFace]:
        try:
            raw_faces = self._app.get(image_bgr)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("InsightFace detection failed: %s", exc)
            return []

        raw_faces = [f for f in raw_faces if float(f.det_score) >= self.detection_threshold]
        raw_faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True
        )
        raw_faces = raw_faces[:max_faces]

        faces: list[DetectedFace] = []
        for idx, face in enumerate(raw_faces):
            keypoints = np.asarray(face.kps, dtype=np.float32).reshape(5, 2)
            aligned, matrix = align_face(image_bgr, keypoints)

            landmarks = getattr(face, "landmark_2d_106", None)
            if landmarks is not None:
                landmarks = np.asarray(landmarks, dtype=np.float32)
                landmarks_aligned = apply_affine(landmarks, matrix)
            else:
                landmarks = keypoints
                landmarks_aligned = apply_affine(keypoints, matrix)

            # InsightFace exposes pose when the pose module is loaded; fall back
            # to solvePnP so behaviour is identical across backends.
            pose_attr = getattr(face, "pose", None)
            if pose_attr is not None and len(pose_attr) == 3:
                pose = (float(pose_attr[1]), float(pose_attr[0]), float(pose_attr[2]))
            else:
                pose = estimate_pose(keypoints, image_bgr.shape[:2])

            faces.append(
                DetectedFace(
                    index=idx,
                    bbox=tuple(float(v) for v in face.bbox),  # type: ignore[arg-type]
                    detection_score=float(face.det_score),
                    keypoints_5=keypoints,
                    aligned=aligned,
                    landmarks=landmarks,
                    landmarks_aligned=landmarks_aligned,
                    pose=pose,
                )
            )
        return faces

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (112, 112):
            aligned_bgr = cv2.resize(aligned_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        if aligned_bgr.dtype != np.uint8:
            aligned_bgr = np.clip(aligned_bgr, 0, 255).astype(np.uint8)
        feature = self._recognition_model.get_feat(aligned_bgr)
        return l2_normalise(np.asarray(feature, dtype=np.float32).ravel())
