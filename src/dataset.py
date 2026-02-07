import torch
from torch.utils.data import Dataset

class NextTokenTripleDataset(Dataset):
    def __init__(self, out_1d: torch.Tensor, tree_1d: torch.Tensor, ctx_1d: torch.Tensor, block_size: int):
        # Assicurati che siano 1D
        assert out_1d.dim() == tree_1d.dim() == ctx_1d.dim() == 1
        assert len(out_1d) == len(tree_1d) == len(ctx_1d)
        self.out = out_1d
        self.tree = tree_1d
        self.ctx = ctx_1d
        self.T = block_size

    def __len__(self):
        return len(self.out) - self.T - 1

    def __getitem__(self, i):
        out_ids  = self.out[i:i+self.T]
        tree_ids = self.tree[i:i+self.T]
        ctx_ids  = self.ctx[i:i+self.T]
        y        = self.out[i+1:i+1+self.T] # next-token target
        return out_ids, tree_ids, ctx_ids, y