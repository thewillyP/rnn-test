import torch
from myrecords import RnnGod

def rnnSplitParameters(rnnGod: RnnGod, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_h = rnnGod.n_h
    n_in = rnnGod.n_in
    n_out = rnnGod.n_out
    w_rec, w_out = torch.split(parameters, [n_h*(n_h+n_in+1), n_out*(n_h+1)])
    return w_rec, w_out

