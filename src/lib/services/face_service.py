from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import onnxruntime
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
        self.model: any = self._load_model(model_path)
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


    def _load_model(self, model_path: Path) -> any:
        mp = Path(model_path)
        if not mp.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        suf = mp.suffix.lower()
        if suf == ".pth":
            return torch.load(mp, map_location="cpu", weights_only=False)
        if suf == ".onnx":
            return onnxruntime.InferenceSession(str(mp))
        raise ValueError(f"Unsupported model format (expected .pth or .onnx): {model_path}")

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (InsightFace / OpenCV convention)
        return image

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Each box is (x1, y1, x2, y2) in pixels (InsightFace convention).
        Return a list of tuples with the coordinates of the faces detected in the image.
        """
        raise NotImplementedError("Not implemented")


    def align_face(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> AlignedFace:
        """
        Crop using box (x1, y1, x2, y2) and run FaceAnalysis on the crop.
        Return an AlignedFace object.
        """
        raise NotImplementedError("Not implemented")

    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Extract embedding from face.
        Return a list of floats representing the embedding of the face.
        """
        raise NotImplementedError("Not implemented")
        
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

        box = faces[0]
        aligned = self.align_face(image, box)
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
        for (x1, y1, x2, y2) in faces:
            aligned = self.align_face(image, (x1, y1, x2, y2))
            embedding = self.extract_embedding_from_face(aligned)
            label, score = self.identify(embedding)
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
