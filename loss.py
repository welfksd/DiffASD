import torch
import os
import torch.nn as nn
import numpy as np
from forward_process import *
import torch.nn.functional as F


def get_loss(model, x_0, t, config):
    x_0 = x_0.to(config.model.device)
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
    b = torch.tensor(betas).type(torch.float).to(config.model.device)
    e = torch.randn_like(x_0, device = x_0.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = at.sqrt() * x_0 + (1- at).sqrt() * e
    output = model(x, t.float())
    loss = F.mse_loss(e, output, reduction="mean")
    # loss = F.mse_loss(e, output, reduction="sum")
    # loss = (e - output).square().mean(dim=(1, 2, 3)).mean(dim=0)
    # loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)  # 最初的loss
    return loss


def get_loss_usingtime(model, x_0, t, cemb, config):
    x_0 = x_0.to(config.model.device)
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.time_steps, dtype=np.float64)
    b = torch.tensor(betas).type(torch.float).to(config.model.device)
    e = torch.randn_like(x_0, device = x_0.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = at.sqrt() * x_0 + (1- at).sqrt() * e
    output = model(x, t.float(), cemb.cuda())
    # loss = F.mse_loss(e, output, reduction="mean")
    # loss = F.mse_loss(e, output, reduction="sum")
    # loss = (e - output).square().mean(dim=(1, 2, 3)).mean(dim=0)
    loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)  # 最初的loss
    return loss


def get_loss_usingtime_github(model, x_0, t, cemb, config):
    x_0 = x_0.to(config.model.device)
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
    b = torch.tensor(betas).type(torch.float).to(config.model.device)
    e = torch.randn_like(x_0, device = x_0.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = at.sqrt() * x_0 + (1- at).sqrt() * e
    output = model(x, t.float(), cemb.cuda())
    # loss = F.mse_loss(e, output, reduction="mean")
    # loss = F.mse_loss(e, output, reduction="sum")
    # loss = (e - output).square().mean(dim=(1, 2, 3)).mean(dim=0)
    loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)  # 最初的loss
    return loss