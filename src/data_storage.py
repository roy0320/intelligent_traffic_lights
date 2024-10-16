from dataclasses import dataclass
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class StoreState:
    """
    Class for storing state variables

    *n_prefix* : current state
    p_* : prime state
    """
    position: torch.Tensor = torch.zeros((1, 4, 4, 16)).to(device)
    speed: torch.Tensor = torch.zeros((1, 4, 4, 16)).to(device)
    tl: torch.Tensor = torch.zeros((1, 1, 4)).to(device)
    p_position: torch.Tensor = torch.zeros((1, 4, 4, 16)).to(device)
    p_speed: torch.Tensor = torch.zeros((1, 4, 4, 16)).to(device)
    p_tl: torch.Tensor = torch.zeros((1, 1, 4)).to(device)

    action: 'typing.Any' = 0
    reward: 'typing.Any' = 0
    done_mask: float = 1.0

    def concat(self):
        self.position = torch.cat(self.position).to(device)
        self.speed = torch.cat(self.speed).to(device)
        self.tl = torch.cat(self.tl).to(device)
        self.p_position = torch.cat(self.p_position).to(device)
        self.p_speed = torch.cat(self.p_speed).to(device)
        self.p_tl = torch.cat(self.p_tl).to(device)

        return (self.position, self.speed, self.tl), \
               (self.p_position, self.p_speed, self.p_tl)

    @property
    def as_tuple(self):
        return (self.position, self.speed, self.tl)

    def swap(self):
        self.position = self.p_position
        self.speed = self.p_speed
        self.tl = self.p_tl

        self.p_position = torch.zeros((1, 4, 4, 16)).to(device)  # depth, rows, cols
        self.p_speed = torch.zeros((1, 4, 4, 16)).to(device)
        self.p_tl = torch.zeros((1, 1, 4)).to(device)
