"""Check UMAP neighbor search when only CPU FAISS is available."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from retrieval.neighbors import faiss_flat_l2_search


class FakeFlatIndex:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def add(self, training):
        self.training = training

    def search(self, queries, k):
        distances = ((queries[:, None, :] - self.training[None, :, :]) ** 2).sum(axis=2)
        indices = np.argsort(distances, axis=1)[:, :k]
        return np.take_along_axis(distances, indices, axis=1), indices


class UmapFaissTests(unittest.TestCase):
    def test_requested_gpu_falls_back_to_cpu_faiss(self):
        cpu_faiss = SimpleNamespace(IndexFlatL2=FakeFlatIndex)
        training = np.array([[0.0], [2.0], [5.0]], dtype=np.float32)
        query = np.array([[1.9]], dtype=np.float32)

        with patch("retrieval.neighbors.faiss", cpu_faiss), self.assertWarnsRegex(
            RuntimeWarning, "using CPU FAISS"
        ):
            distances, indices = faiss_flat_l2_search(
                training, query, k=2, return_distance=True, gpu_id=0
            )

        self.assertEqual(indices.tolist(), [[1, 0]])
        np.testing.assert_allclose(distances, [[0.01, 3.61]], rtol=1e-5)

    def test_null_gpu_id_uses_cpu_without_warning(self):
        cpu_faiss = SimpleNamespace(IndexFlatL2=FakeFlatIndex)
        with patch("retrieval.neighbors.faiss", cpu_faiss), patch(
            "retrieval.neighbors.warnings.warn"
        ) as warn:
            indices = faiss_flat_l2_search(
                [[0.0], [2.0]], [[1.9]], k=1, gpu_id=None
            )
        self.assertEqual(indices.tolist(), [[1]])
        warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
