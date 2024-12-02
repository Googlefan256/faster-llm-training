from torch import nn
import torch


class Test(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, a, b):
        return a + b


test = Test()
a = torch.full([1], 0.1, dtype=torch.bfloat16, device="cuda:0",requires_grad=True)
b = torch.full([1], 0.1, dtype=torch.bfloat16, device="cuda:0",requires_grad=True)

for i in range(1_000_00):
    res: torch.Tensor = test(a, b)
    res.backward()
