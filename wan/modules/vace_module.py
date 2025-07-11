import torch
from contextlib import contextmanager
import torch.distributed as dist

class OffloadManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0

    def offload_modules(self):
        for block in getattr(self.model, 'blocks', []):
            block.to('cpu')
        for block in getattr(self.model, 'vace_blocks', []):
            block.to('cpu')
        torch.cuda.empty_cache()

    @contextmanager
    def onload_block(self, block_type, idx):
        if block_type == 'main':
            block = self.model.blocks[idx]
        elif block_type == 'vace':
            block = self.model.vace_blocks[idx]
        else:
            raise ValueError('Invalid block type')
        block.to(self.device)
        try:
            yield
        finally:
            block.to('cpu')