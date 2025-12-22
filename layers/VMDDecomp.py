from torch import nn
import torch
import torch.nn.functional as F


class VMDDecomp(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.K = getattr(configs, 'vmd_scales', 5)
        self.alpha = getattr(configs, 'vmd_alpha', 2000)
        self.tau = getattr(configs, 'vmd_tau', 0.0)
        self.DC = getattr(configs, 'vmd_dc', 0)
        self.tol = getattr(configs, 'vmd_tol', 1e-5)  
        self.n_iter = getattr(configs, 'vmd_n_iter', 100)  
        
        self.cache = {}
        
    def _get_or_create_cache(self, T, device):
        cache_key = (T, device)
        if cache_key not in self.cache:
            if self.DC:
                omega = torch.zeros(self.K, device=device)
                omega[1:] = torch.linspace(0.05, 0.45, self.K-1, device=device)
            else:
                omega = torch.linspace(0.05, 0.45, self.K, device=device)
            
            freqs = torch.fft.rfftfreq(T, device=device)
            N_half = len(freqs)
            
            self.cache[cache_key] = {
                'omega': omega.clone(),
                'freqs': freqs,
                'N_half': N_half
            }
        
        return self.cache[cache_key]
    
    def _vmd_batch(self, signal):
    
        B, T, M = signal.shape
        device = signal.device

        cache = self._get_or_create_cache(T, device)
        omega = cache['omega'].clone() 
        freqs = cache['freqs'] 
        N_half = cache['N_half']

        signal = signal.permute(0, 2, 1).reshape(B * M, T)  
        f_hat = torch.fft.rfft(signal, dim=1)  
        f_hat = f_hat.reshape(B, M, N_half).permute(0, 2, 1)  

        u_hat = torch.zeros(B, N_half, M, self.K, dtype=torch.complex64, device=device)
        lambda_hat = torch.zeros(B, N_half, M, dtype=torch.complex64, device=device)

        alpha_freq_sq = self.alpha * freqs.view(1, -1, 1, 1) ** 2  

        for n in range(self.n_iter):
            u_hat_old = u_hat
            for k in range(self.K):
                sum_uk = u_hat.sum(dim=-1) - u_hat[..., k]  
                numerator = f_hat - sum_uk - lambda_hat / 2

                freq_diff_sq = (freqs.view(1, -1, 1) - omega[k]) ** 2  
                denominator = 1 + alpha_freq_sq[..., 0] * freq_diff_sq + 1e-8
                
                u_hat[..., k] = numerator / denominator

            if n % 5 == 0:  
                start_k = 1 if self.DC else 0
                u_power = torch.abs(u_hat) ** 2 
                
                for k in range(start_k, self.K):
                    power_k = u_power[..., k] 
                    weighted = freqs.view(1, -1, 1) * power_k
                    
                    omega[k] = torch.clamp(
                        weighted.sum() / (power_k.sum() + 1e-8),
                        0.01, 0.49
                    ).detach()
            
            lambda_hat += self.tau * (u_hat.sum(dim=-1) - f_hat)

            if n % 10 == 0 and n > 0:
                with torch.no_grad():
                    err = (u_hat - u_hat_old).norm() / (u_hat_old.norm() + 1e-8)
                    if err < self.tol:
                        break
        
        u_complex = torch.fft.irfft(u_hat.permute(0, 3, 2, 1), n=T, dim=-1)  
        u = u_complex.permute(0, 3, 2, 1).real  
        
        return u, omega
    
    def forward(self, x):
        B, L, M = x.shape
        
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        u, omega = self._vmd_batch(x)  # [B, L, M, K]

        with torch.no_grad():
            freq_order = torch.argsort(omega)
        u = u[..., freq_order]

        return [u[..., k] for k in range(self.K)]