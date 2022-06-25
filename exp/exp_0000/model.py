from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim  # (1, 4, 4)

        self.fc = nn.Sequential(
            nn.Linear(c*h*w, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.values = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.advantages = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        values = self.values(x)
        advantages = self.advantages(x)
        q = values + (advantages - advantages.mean(dim=1, keepdims=True))
        return q
