import sys
import cv2
from pathlib import Path
from src.lib.services.face_service import FaceService
from src.lib.storage.pgvector_store import PgVectorEmbeddingStore
import torch
import onnxruntime
import numpy as np

model_path = Path("models/face_detection.onnx")
class DummyStore:
    def all(self): return []
    def append(self, r): pass

fs = FaceService(
    store=DummyStore(),
    similarity_metric="cosine",
    similarity_threshold=0.55,
    face_size=112,
    model_path=model_path,
    output_path=Path("output")
)

# Create a dummy image
img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
cv2.imwrite("test_face.jpg", img)

# Actually, let's just create random aligned faces
class DummyFace:
    def __init__(self, image):
        self.image = image

aligned1 = DummyFace(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
aligned2 = DummyFace(aligned1.image.copy()) # Exact same image

emb1 = fs.extract_embedding_from_face(aligned1)
emb2 = fs.extract_embedding_from_face(aligned2)

sim = fs.similarity(emb1, emb2)
print("Similarity between identical images:", sim)

# add slight noise
aligned3 = DummyFace(np.clip(aligned1.image.astype(int) + np.random.randint(-10, 10, (112, 112, 3)), 0, 255).astype(np.uint8))
emb3 = fs.extract_embedding_from_face(aligned3)
sim2 = fs.similarity(emb1, emb3)
print("Similarity with noise:", sim2)

# completely different image
aligned4 = DummyFace(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
emb4 = fs.extract_embedding_from_face(aligned4)
sim3 = fs.similarity(emb1, emb4)
print("Similarity between different images:", sim3)

