import sys
import torch
import numpy as np
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from quantization.rqkmeans import KMeans, RQKMeans
from quantization.utils import disambiguate_codes


def test_kmeans_fit_predict_cpu():
    torch.manual_seed(0)
    X1 = torch.randn(20, 2) * 0.05 + torch.tensor([2.0, 2.0])
    X2 = torch.randn(20, 2) * 0.05 + torch.tensor([-2.0, -2.0])
    X = torch.cat([X1, X2], dim=0)

    km = KMeans(n_clusters=2, batch_size=8, max_iter=10, tol=1e-4, verbose=0, random_state=123, gpu_preload=False, device="cpu")
    km.fit(X)
    y = km.predict(X)
    assert y.shape == (40,)
    uniq = torch.unique(y)
    assert uniq.numel() == 2


def test_kmeans_predict_before_fit_raises():
    X = torch.randn(5, 3)
    km = KMeans(n_clusters=2, batch_size=4, max_iter=1, verbose=0, random_state=1, gpu_preload=False, device="cpu")
    with pytest.raises(RuntimeError):
        _ = km.predict(X)


def test_kmeans_more_clusters_than_samples_raises():
    X = torch.randn(3, 2)
    km = KMeans(n_clusters=5, batch_size=4, max_iter=1, verbose=0, random_state=1, gpu_preload=False, device="cpu")
    with pytest.raises(AssertionError):
        km.fit(X)


def test_rqkmeans_fit_predict_cpu():
    torch.manual_seed(0)
    X = torch.randn(32, 4)
    rq = RQKMeans(num_clusters=3, num_ids=2, batch_size=8, max_iter=10, tol=1e-4, verbose=0, random_state=123, gpu_preload=False, device="cpu")
    rq.fit(X)
    codes = rq.predict(X)
    assert codes.shape == (32, 2)
    assert torch.all((codes >= 0) & (codes < 4))


def test_disambiguate_codes_basic():
    codes = np.array([[1, 2], [1, 2], [0, 0]], dtype=np.int64)
    ids = ["A", "B", "C"]
    codes_aug, suffix, stats = disambiguate_codes(codes, ids, order="by_id")
    assert codes_aug.shape[1] == 3
    assert set(suffix[:2]) == {0, 1}
    assert stats["levels_after"] == 3


def test_disambiguate_codes_no_collision():
    codes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    ids = ["A", "B"]
    _, suffix, stats = disambiguate_codes(codes, ids, order="by_id")
    assert np.all(suffix == 0)
    assert stats["total_collided_items"] == 0
    assert stats["levels_after"] == codes.shape[1] + 1


def test_disambiguate_codes_order_by_index():
    codes = np.array([[7, 7], [7, 7], [0, 1]], dtype=np.int64)
    ids = ["Z2", "Z1", "X"]
    suffix = disambiguate_codes(codes, ids, order="by_index")[1]
    assert suffix[0] == 0 and suffix[1] == 1
