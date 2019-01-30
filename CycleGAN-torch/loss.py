import torch

def real_mse_loss(D_out):
    return torch.mean((D_out-1)**2)

def fake_mse_loss(D_out):
    return torch.mean(D_out**2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):

    cycle_consistency_loss = lambda_weight*torch.mean(torch.abs(real_im-reconstructed_im))
    return cycle_consistency_loss
