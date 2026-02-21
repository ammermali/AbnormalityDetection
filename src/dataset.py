import random
import torch
from torch.utils.data import Dataset

import random
import torch
from torch.utils.data import Dataset

class TxRandomWindowDataset(Dataset):
    def __init__(self, out_seqs, tree_seqs, ctx_seqs, block_size: int, num_samples_per_epoch: int):
        assert isinstance(out_seqs, list) and isinstance(tree_seqs, list) and isinstance(ctx_seqs, list)
        assert len(out_seqs) == len(tree_seqs) == len(ctx_seqs)

        self.out_seqs = out_seqs
        self.tree_seqs = tree_seqs
        self.ctx_seqs = ctx_seqs
        self.T = block_size
        self.num_samples_per_epoch = num_samples_per_epoch

        self.valid_j = []
        for j in range(len(out_seqs)):
            o, tr, cx = out_seqs[j], tree_seqs[j], ctx_seqs[j]
            assert torch.is_tensor(o) and o.dim() == 1
            assert torch.is_tensor(tr) and tr.dim() == 1
            assert torch.is_tensor(cx) and cx.dim() == 1
            assert len(o) == len(tr) == len(cx)
            if len(o) > self.T + 1:
                self.valid_j.append(j)

        if not self.valid_j:
            raise ValueError("No sequences long enough for block_size.")

    def __len__(self):
        return self.num_samples_per_epoch

    def __getitem__(self, _):
        j = random.choice(self.valid_j)
        out = self.out_seqs[j]
        tree = self.tree_seqs[j]
        ctx = self.ctx_seqs[j]

        start = random.randint(0, len(out) - self.T - 2)

        x_out  = out[start:start+self.T]
        x_tree = tree[start:start+self.T]
        x_ctx  = ctx[start:start+self.T]
        y      = out[start+1:start+1+self.T]
        return x_out, x_tree, x_ctx, y


class TxStrideWindowDataset(Dataset):
    def __init__(self, out_seqs, tree_seqs, ctx_seqs, block_size: int, stride: int = None):
        assert len(out_seqs) == len(tree_seqs) == len(ctx_seqs)
        self.out_seqs = out_seqs
        self.tree_seqs = tree_seqs
        self.ctx_seqs = ctx_seqs
        self.T = block_size
        self.stride = stride if stride is not None else block_size

        self.samples = []
        for j, out in enumerate(out_seqs):
            if len(out) <= self.T + 1:
                continue
            for start in range(0, len(out) - self.T - 1, self.stride):
                self.samples.append((j, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        j, start = self.samples[i]
        out = self.out_seqs[j]
        tree = self.tree_seqs[j]
        ctx = self.ctx_seqs[j]

        x_out  = out[start:start+self.T]
        x_tree = tree[start:start+self.T]
        x_ctx  = ctx[start:start+self.T]
        y      = out[start+1:start+1+self.T]
        return x_out, x_tree, x_ctx, y
