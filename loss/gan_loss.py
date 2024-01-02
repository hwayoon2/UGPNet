from basicsr.losses.losses import r1_penalty
from torch.nn import functional as F


def cal_adv_d_loss(fake_d_pred, real_d_pred):
    real_loss = F.softplus(-real_d_pred).mean()
    fake_loss = F.softplus(fake_d_pred).mean()
    D_loss = real_loss + fake_loss
    return D_loss

def regd(real, real_pred):
    r1_reg_weight = 10
    net_d_iters = 1
    net_d_init_iters = 0
    net_d_reg_every = 10
    l_d_r1 = r1_penalty(real_pred, real)
    l_d_r1 = (r1_reg_weight / 2 * l_d_r1 * net_d_reg_every + 0 * real_pred[0])
    return l_d_r1

def cal_adv_loss(fake_g_pred):
    adv_loss = F.softplus(-fake_g_pred).mean()
    return adv_loss
