from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from lib.schemas import EmbeddingRecord, FaceDetection, PredictResult, AlignedFace
from lib.storage.base import EmbeddingStoreProtocol
import os 
import logging

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        face_size: int,
        model_path: Path,
        output_path: Path = Path("output"),
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.face_size = face_size
        self.model = None # Custom model replaced by InsightFace
        self.output_path = output_path

        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    @staticmethod
    def _kps_to_keypoints_dict(kps: np.ndarray | None) -> dict[str, list[int]]:
        if kps is None or len(kps) == 0:
            return {}
        return {
            f"k{i}": [int(round(float(kps[i, 0]))), int(round(float(kps[i, 1])))]
            for i in range(len(kps))
        }


    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (InsightFace / OpenCV convention)
        return image

    def detect_faces(self, image: np.ndarray) -> list[any]:
        """
        Detect faces in the image and return the full InsightFace objects.
        This preserves landmarks for the alignment step.
        """
        if not hasattr(self, "_app"):
            self._app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self._app.prepare(ctx_id=0)
            
        faces = self._app.get(image)
        # Prioritize the largest detected face (sorting by area)
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces

    def align_face(
        self, image: np.ndarray, face_obj: any
    ) -> AlignedFace:
        """
        Align face using the provided detection object.
        Using the landmarks already found during detection prevents fallback to cv2.resize.
        """

        box = tuple(map(int, face_obj.bbox))
        x1, y1, x2, y2 = box
        
        # Predefined size to maintain consistency with training (112x112)
        model_input_size = 112 

        if face_obj is not None and face_obj.kps is not None:
            # High-quality alignment (Similarity Transform)
            aligned_bgr = face_align.norm_crop(image, landmark=face_obj.kps, image_size=model_input_size)
            
            # If norm_crop fails, we still try to at least return the resized crop 
            # but we log it as it's sub-optimal for recognition.
            if aligned_bgr is None or aligned_bgr.size == 0:
                logger.warning("norm_crop returned empty image, falling back to resize.")
                h, w = image.shape[:2]
                cx1, cy1, cx2, cy2 = self._clip_xyxy(x1, y1, x2, y2, h, w)
                aligned_bgr = cv2.resize(image[cy1:cy2, cx1:cx2], (model_input_size, model_input_size))
            
            kps_adj = face_obj.kps.copy()
            kps_adj[:, 0] -= x1
            kps_adj[:, 1] -= y1
        else:
            # This should rarely happen now that we pass the detection object directly
            logger.error(f"Alignment failed: No keypoints found for face at {box}. Using low-quality resize.")
            h, w = image.shape[:2]
            cx1, cy1, cx2, cy2 = self._clip_xyxy(x1, y1, x2, y2, h, w)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                aligned_bgr = np.zeros((model_input_size, model_input_size, 3), dtype=np.uint8)
            else:
                aligned_bgr = cv2.resize(crop, (model_input_size, model_input_size))
            kps_adj = None
            
        embedding = face_obj.normed_embedding.tolist() if face_obj is not None and face_obj.normed_embedding is not None else None
        return AlignedFace(bbox=box, keypoints=kps_adj, image=aligned_bgr, embedding=embedding)

    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Extract embedding from face using InsightFace's recognition model.
        """
        if face.embedding is not None:
            return face.embedding

        # Fallback: if embedding is missing, compute it from the aligned image
        if not hasattr(self, "_app"):
            self._app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self._app.prepare(ctx_id=0)

        # InsightFace recognition models expect BGR 112x112 images.
        # We can reach into the loaded models to run recognition directly on the aligned image.
        if 'recognition' in self._app.models:
            # face.image is already BGR and 112x112 from align_face
            embedding = self._app.models['recognition'].get(face.image, face)
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            return embedding
        
        logger.error("InsightFace recognition model not found in FaceAnalysis app.")
        return []
        
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    def identify(self, query_embedding: list[float]) -> tuple[str, float]:
        records = self.store.all()
        if not records:
            return "unknown", 0.0

        best_label = "unknown"
        best_score = -1.0
        for record in records:
            score = self.similarity(query_embedding, record.embedding)
            if score > best_score:
                best_score = score
                best_label = record.etiqueta

        if best_score < self.similarity_threshold:
            return "unknown", max(best_score, 0.0)
        return best_label, best_score

    def register_identity(
        self, identity: str, image_path: str, metadata: dict[str, object]
    ) -> EmbeddingRecord:
        image = self._load_image(image_path)
        faces = self.detect_faces(image)

        if len(faces) != 1:
            raise ValueError("Exactly one face must be detected for identity registration.")
        
        logger.info(f"Face detected: {faces[0]}")

        face_obj = faces[0]
        if face_obj.kps is None:
            raise ValueError("No landmarks detected. High-quality alignment is required for identity registration.")
            
        aligned = self.align_face(image, face_obj)
        embedding = self.extract_embedding_from_face(aligned)

        img_id = str(uuid4())
        img_output_path = self.output_path / f"img_{img_id}.jpg"
        
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(img_output_path),
            etiqueta=identity,
            metadata=metadata,
        )
        self.store.append(record)

        cv2.imwrite(str(img_output_path), aligned.image)
        logger.info(f"Identity registered: {identity} with image: {image_path}")
        return record

    def predict(self, source_path: str, output_path: Path) -> str:
        image = self._load_image(source_path)
        faces = self.detect_faces(image)
        detections: list[FaceDetection] = []
        for face_obj in faces:
            aligned = self.align_face(image, face_obj)
            embedding = self.extract_embedding_from_face(aligned)
            label, score = self.identify(embedding)
            
            x1, y1, x2, y2 = map(int, face_obj.bbox)
            kps = getattr(aligned, "keypoints", None)
            kps_arr = np.asarray(kps) if kps is not None else None
            detections.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    keypoints=self._kps_to_keypoints_dict(kps_arr),
                    label=label,
                    score=round(float(score), 4),
                )
            )

        detected_people = sorted({item.label for item in detections if item.label != "unknown"})
        result_payload = PredictResult(
            source_path=source_path,
            detections=detections,
            detected_people=detected_people,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(result_payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)