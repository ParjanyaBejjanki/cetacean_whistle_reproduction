"""Losses of eqs. 3-5, implemented literally (specification §3.2).

The critic output of the paper is D = sigmoid(a) (Table 8). The WGAN-GP critic loss (eq. 5) and its gradient penalty
are computed on that POST-SIGMOID output, and the adversarial term of the VAE (eq. 4) is -log D(x_hat). This differs
from textbook WGAN-GP (unbounded critic, generator loss -E[D]); the paper's formulation is reproduced, not replaced.
log D is evaluated as logsigmoid(a), which is the same function without log(0) when sigmoid(a) underflows.

Every loss is a batch MEAN of per-image terms (the expectations of eqs. 3-5); within an image the reconstruction
term is a SUM over pixels and the KL term a SUM over latent dimensions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def recon_sse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per-image sum of squared errors, shape (B,). [DECISION] Gaussian NLL up to a constant."""
    return (x_hat - x).pow(2).flatten(1).sum(1)


def kl_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, diag exp(logvar)) || N(0, I)) per image, summed over dims, shape (B,)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(1)


def adv_neg_log_d(critic_logit: torch.Tensor) -> torch.Tensor:
    """-log D(x_hat) per image with D = sigmoid(logit) (eq. 4)."""
    return -F.logsigmoid(critic_logit)


def vae_loss(x_hat, x, mu, logvar, critic_logit_fake, beta: float, lambda_adv: float, use_critic: bool = True):
    """Eq. 4: L = L_rec + beta L_KL + lambda L_adv (batch means). Returns (total, parts dict of detached means)."""
    rec = recon_sse(x_hat, x).mean()
    kl = kl_std_normal(mu, logvar).mean()
    total = rec + beta * kl
    parts = {"rec": rec.detach(), "kl": kl.detach()}
    if use_critic:
        adv = adv_neg_log_d(critic_logit_fake).mean()
        total = total + lambda_adv * adv
        parts["adv"] = adv.detach()
    return total, parts


def critic_loss(critic, x_real: torch.Tensor, x_fake: torch.Tensor, eps: torch.Tensor, lambda_gp: float):
    """Eq. 5 on D = sigmoid(logit):  E[D(x_hat)] - E[D(x)] + lambda_GP E[(||grad_x~ D(x~)||_2 - 1)^2],
    x~ = eps x + (1 - eps) x_hat, eps ~ U(0, 1) per image (supplied by the caller, shape (B, 1, 1, 1))."""
    d_real = critic(x_real)
    d_fake = critic(x_fake)
    x_t = (eps * x_real + (1.0 - eps) * x_fake).requires_grad_(True)
    d_t = critic(x_t)
    (grad,) = torch.autograd.grad(d_t.sum(), x_t, create_graph=True)
    gnorm = grad.flatten(1).norm(2, dim=1)
    gp = (gnorm - 1.0).pow(2).mean()
    loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
    return loss, {"d_real": d_real.mean().detach(), "d_fake": d_fake.mean().detach(), "gp": gp.detach(),
                  "gnorm": gnorm.mean().detach()}
