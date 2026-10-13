import torch
import torch.nn as nn


class ShallowMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_size = 16
    batch_size = 8

    model = ShallowMLP(input_size).to(device)
    x = torch.randn(batch_size, input_size, device=device)

    logits = model(x).squeeze(1)          # (batch_size,)
    policy = torch.softmax(logits, dim=0) # probability distribution over the batch

    print(f"Input shape:  {x.shape}")
    print(f"Logits:  {logits}")
    print(f"Policy:  {policy}")
    print(f"Sums to: {policy.sum():.6f}")
    print(f"Selected action: {policy.argmax().item()}")
