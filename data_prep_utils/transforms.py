import torch


class StackLabel:
    def __call__(self, label):
        return torch.tensor([int(label)]).to(torch.int64)
