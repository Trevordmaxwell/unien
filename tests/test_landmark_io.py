import torch

from uelm4.memory.landmark_io import load_landmarks, save_landmarks


def test_landmark_io_roundtrip(tmp_path):
    tensor = torch.randn(8, 4)
    tensor_path, meta_path = save_landmarks(tensor, tmp_path / "landmarks")
    loaded = load_landmarks(meta_path)
    assert torch.allclose(loaded, tensor)
    assert tensor_path.exists()
    assert meta_path.exists()
