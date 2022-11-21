from torch import Tensor
import typing as tp

class ConstrastiveTrans:
    def __init__(self, base_transforms, n_trans: int = 2) -> None:
        self.bases = base_transforms
        self.n_views = n_trans

    def __call__(self, x) -> tp.List[Tensor]:
        return [self.bases(x) for view in range(self.n_views)]