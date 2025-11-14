# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tfunc

def make_beta_schedule(schedule, num_timesteps, beta_start=1e-4, beta_end=2e-2):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule == "quadratic":
        return torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif schedule == "cosine":
        s = 8e-3
        timesteps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
        alphas_cumprod = torch.cos((timesteps + s) / (1+s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 1e-6, 0.9999).float()
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule}")

class DiffusionProcess:
    def __init__(self, num_time_steps, schedule_type="linear", use_v_param=True):
        self.T = num_time_steps
        self.betas = make_beta_schedule(schedule_type, self.T)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_oneminus_alpha_cumprod = torch.sqrt(1-self.alpha_cumprod)
        self.alpha_cumprod_prev = tfunc.pad(self.alpha_cumprod[:-1],(1,0),value=1.0)
        self.posterior_variance = self.betas*(1-self.alpha_cumprod_prev)/(1 - self.alpha_cumprod)
        self.use_v_param = use_v_param
    def _extract(self, a, t, shape):
        out = a.gather(-1, t).view(t.shape[0],*(1,)*(len(shape)-1))
        return out
    def x0_from_eps(self, xt, t, eps):
        a_bar_t = self._extract(self.sqrt_alpha_cumprod, t, xt.shape)
        sigma = self._extract(self.sqrt_oneminus_alpha_cumprod, t, xt.shape)
        x0 = (xt - sigma * eps) / torch.clamp(a_bar_t,1e-7)
        return x0
    def x0_from_v(self, xt, t, v):
        a_bar_t = self._extract(self.sqrt_alpha_cumprod, t, xt.shape)
        sigma = self._extract(self.sqrt_oneminus_alpha_cumprod, t, xt.shape)
        x0 = a_bar_t * xt - sigma * v
        return x0
    def eps_from_v(self, xt, t, v):
        a_bar_t = self._extract(self.sqrt_alpha_cumprod, t, xt.shape)
        sigma = self._extract(self.sqrt_oneminus_alpha_cumprod, t, xt.shape)
        eps = sigma * xt + a_bar_t * v
        return eps
    def backward_process_compute(self, eps_theta, xt, t, return_mean=False):
        betas_t = self._extract(self.betas, t, xt.shape)
        a_t = self._extract(self.alphas, t, xt.shape)
        a_bar_t = self._extract(self.alpha_cumprod, t, xt.shape)
        mean = (1/torch.sqrt(a_t))*(xt -(betas_t/torch.sqrt(1 - a_bar_t))*eps_theta)
        if (t==0).all() or return_mean :
            return mean
        var = self._extract(self.posterior_variance, t, xt.shape)
        noise = torch.randn_like(xt)
        return mean + torch.sqrt(var)*noise

    @torch.no_grad()
    def backward_process(self, model, xt, t):
        if self.use_v_param:
            v_pred = model(xt, t)
            eps_theta = self.eps_from_v(xt, t, v_pred)
        else:            
            eps_theta = model(xt, t)
        return self.backward_process_compute(eps_theta, xt, t)
    @torch.no_grad()
    def backward_process_loop(self, model, shape):
        img = torch.randn(shape, device=next(model.parameters()).device) 
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=img.device)
            img = self.backward_process(model, img, t)
        return img
    
    def forward_process(self, x0, t):
        a_bar_t = self._extract(self.sqrt_alpha_cumprod, t, x0.shape)
        sigma = self._extract(self.sqrt_oneminus_alpha_cumprod, t, x0.shape)
        noise = torch.randn_like(x0)
        xt = a_bar_t * x0 + sigma * noise
        return xt, noise, a_bar_t, sigma
    
    def forward_loss(self, model, x0, t, loss_fn):
        xt, noise, a_bar_t, sigma = self.forward_process(x0, t)
        if self.use_v_param:
            v_target = a_bar_t * noise - sigma * x0
            v_theta = model(xt, t)
            x0_pred = self.x0_from_v(xt, t, v_theta)
            return loss_fn(v_target, v_theta, x0, x0_pred, t)
        else:
            eps_theta = model(xt, t)
            x0_pred = self.x0_from_eps(xt, t, eps_theta)
            return loss_fn(noise, eps_theta, x0, t)

if __name__ == "__main__":
    import cv2
    from matplotlib import pyplot as plt
    noise_schedule = np.linspace(0, 0.02, 100)
    diffusion = DiffusionProcess(noise_schedule)
    
    # Example usage
    x0 =  cv2.imread(r"""C:\Users\rkvai\OneDrive\Pictures\Screenshots\Screenshot 2025-07-13 135001.png""")/255. # Example input image
    t = 50  # Example time step
    xt, noise = diffusion.forward_process(x0, t)
    x0_reconstructed = diffusion.backward_process(xt, noise, t)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(x0)    
    ax[1].imshow(x0_reconstructed)
    ax[0].set_title("Original Image")
    ax[1].set_title("Reconstructed Image")
    plt.show()
    assert np.allclose(x0, x0_reconstructed), "Reconstruction failed"
    print("Original shape:", x0.shape)
    print("Noisy shape:", xt.shape)
    print("Reconstructed shape:", x0_reconstructed.shape)