import os
import tempfile
import uuid
import pytest
import torch.distributed as dist

def init_dist_once():
    if dist.is_initialized():
        return
    init_method = f"file://{os.path.join(tempfile.gettempdir(), 'ptdist' + str(uuid.uuid4()))}"
    dist.init_process_group(backend="gloo", init_method=init_method, rank=0, world_size=1)

@pytest.fixture(scope="session", autouse=True)
def init_dist_session():
    init_dist_once()
    yield
    if dist.is_initialized():
        dist.destroy_process_group()

@pytest.fixture(scope="session")
def tmp_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("outputs")
