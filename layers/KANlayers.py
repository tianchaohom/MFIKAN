import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List
import numpy as np


class DynamicSparseMask:
    def __init__(self, shape: tuple, initial_sparsity: float = 0.0, device: str = 'cpu'):
        self.shape = shape
        self.device = device
        self.current_sparsity = initial_sparsity

        initial_active = int((1 - initial_sparsity) * np.prod(shape))
        self.mask = torch.zeros(shape, dtype=torch.bool, device=device)
        if initial_active > 0:
            flat_indices = torch.randperm(np.prod(shape))[:initial_active]
            self.mask.view(-1)[flat_indices] = True

        self.importance_scores = torch.zeros(shape, device=device)
        self.gradient_scores = torch.zeros(shape, device=device)

        self.activation_history = torch.zeros(shape, device=device)
        self.pruning_history = torch.zeros(shape, device=device)

    def update_importance_scores(self, weights: torch.Tensor, gradients: torch.Tensor = None):
        self.importance_scores = torch.abs(weights.detach())

        if gradients is not None:
            self.gradient_scores = torch.abs(gradients.detach())
            self.importance_scores = 0.7 * self.importance_scores + 0.3 * self.gradient_scores

    def prune_connections(self, prune_ratio: float, strategy: str = 'magnitude'):
        if prune_ratio <= 0:
            return 0

        current_active = self.mask.sum().item()
        num_to_prune = int(current_active * prune_ratio)

        if num_to_prune == 0:
            return 0

        if strategy == 'magnitude':
            active_scores = self.importance_scores[self.mask]
            if len(active_scores) > 0:
                threshold = torch.kthvalue(active_scores, num_to_prune)[0]
                prune_mask = (self.importance_scores <= threshold) & self.mask
                prune_indices = torch.nonzero(prune_mask, as_tuple=False)
                if len(prune_indices) > num_to_prune:
                    prune_values = self.importance_scores[prune_mask]
                    _, sorted_indices = torch.sort(prune_values, descending=False)
                    prune_indices = prune_indices[sorted_indices[:num_to_prune]]

                for idx in prune_indices:
                    self.mask[tuple(idx)] = False
                    self.pruning_history[tuple(idx)] += 1

        elif strategy == 'random':
            active_indices = torch.nonzero(self.mask, as_tuple=False)
            if len(active_indices) > 0:
                prune_indices = active_indices[torch.randperm(len(active_indices))[:num_to_prune]]
                for idx in prune_indices:
                    self.mask[tuple(idx)] = False
                    self.pruning_history[tuple(idx)] += 1

        return num_to_prune

    def grow_connections(self, grow_ratio: float, strategy: str = 'gradient', weights: torch.Tensor = None):
        if grow_ratio <= 0:
            return 0

        current_inactive = (~self.mask).sum().item()
        num_to_grow = int(current_inactive * grow_ratio)

        if num_to_grow == 0:
            return 0

        grow_indices = []
        if strategy == 'gradient':
            inactive_scores = self.gradient_scores[~self.mask]
            if len(inactive_scores) > 0:
                k_val = max(1, len(inactive_scores) - num_to_grow + 1)
                if k_val > len(inactive_scores):
                    k_val = len(inactive_scores)
                
                threshold = torch.kthvalue(inactive_scores, k_val).values
                grow_mask = (self.gradient_scores >= threshold) & (~self.mask)
                
                potential_grow_indices = torch.nonzero(grow_mask, as_tuple=False)
                if len(potential_grow_indices) > num_to_grow:
                    scores_of_potential = self.gradient_scores[grow_mask]
                    _, top_indices_in_potential = torch.topk(scores_of_potential, num_to_grow)
                    grow_indices = potential_grow_indices[top_indices_in_potential]
                else:
                    grow_indices = potential_grow_indices
            
        elif strategy == 'random':
            inactive_indices = torch.nonzero(~self.mask, as_tuple=False)
            if len(inactive_indices) > 0:
                grow_indices = inactive_indices[torch.randperm(len(inactive_indices))[:num_to_grow]]

        elif strategy == 'importance':
            importance_potential = self.activation_history - self.pruning_history
            inactive_scores = importance_potential[~self.mask]
            if len(inactive_scores) > 0:
                k_val = max(1, len(inactive_scores) - num_to_grow + 1)
                if k_val > len(inactive_scores):
                    k_val = len(inactive_scores)

                threshold = torch.kthvalue(inactive_scores, k_val).values
                grow_mask = (importance_potential >= threshold) & (~self.mask)
                
                potential_grow_indices = torch.nonzero(grow_mask, as_tuple=False)
                if len(potential_grow_indices) > num_to_grow:
                    scores_of_potential = importance_potential[grow_mask]
                    _, top_indices_in_potential = torch.topk(scores_of_potential, num_to_grow)
                    grow_indices = potential_grow_indices[top_indices_in_potential]
                else:
                    grow_indices = potential_grow_indices

        for idx in grow_indices:
            self.mask[tuple(idx)] = True
            self.activation_history[tuple(idx)] += 1

            if weights is not None:
                weights[tuple(idx)] = torch.randn(1, device=self.device) * 0.01
        
        return len(grow_indices)

    def get_sparsity(self) -> float:
        return 1.0 - (self.mask.sum().item() / self.mask.numel())

    def apply_mask(self, weights: torch.Tensor) -> torch.Tensor:
        return weights * self.mask.float()


class BSplineActivation(nn.Module):
    def __init__(self,
                 grid_size: int = 20,  
                 spline_order: int = 6,
                 grid_min: float = -3.0, 
                 grid_max: float = 3.0,
                 noise_scale: float = 0.1,
                 initial_sparsity: float = 0.0,
                 target_sparsity: float = 0.1,
                 dst_schedule: str = 'cosine'):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.dst_schedule = dst_schedule

        self.register_buffer('grid_min', torch.tensor(grid_min, dtype=torch.float32))
        self.register_buffer('grid_max', torch.tensor(grid_max, dtype=torch.float32))

        self.n_coeffs = grid_size + spline_order + 1
        self.noise_scale = noise_scale

        self.fine_grid_size = grid_size * 2
        self.coarse_grid_size = max(grid_size // 2, 5)

        self.register_buffer('input_stats', torch.zeros(2))
        self.register_buffer('update_count', torch.tensor(0))

        self.spline_weights = None
        self.linear_weights = None
        self.input_dim = None
        self.dynamic_mask = None

        self.fine_weights = None
        self.coarse_weights = None

        self.attention_weights = None

        self.register_buffer('training_step', torch.tensor(0))

        self._create_control_points()

    def _create_control_points(self):
        control_points = torch.linspace(
            self.grid_min, self.grid_max, self.n_coeffs,
            dtype=torch.float32,
            device=self.grid_min.device  
        )
        self.register_buffer('control_points', control_points)
        fine_n_coeffs = self.fine_grid_size + self.spline_order + 1
        fine_control_points = torch.linspace(
            self.grid_min, self.grid_max, fine_n_coeffs,
            dtype=torch.float32,
            device=self.grid_min.device  
        )
        self.register_buffer('fine_control_points', fine_control_points)
        coarse_n_coeffs = self.coarse_grid_size + self.spline_order + 1
        coarse_control_points = torch.linspace(
            self.grid_min, self.grid_max, coarse_n_coeffs,
            dtype=torch.float32,
            device=self.grid_min.device  
        )
        self.register_buffer('coarse_control_points', coarse_control_points)

        sigma = (self.grid_max - self.grid_min) / self.grid_size
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32, device=self.grid_min.device))
        self.register_buffer('inv_sigma_sq', torch.tensor(1.0 / (2 * sigma**2), dtype=torch.float32, device=self.grid_min.device))

    def _initialize_weights(self, input_dim: int, device: torch.device):
        if self.spline_weights is None or self.input_dim != input_dim:
            self.input_dim = input_dim

            self.spline_weights = nn.Parameter(
                torch.randn(input_dim, self.n_coeffs, dtype=torch.float32, device=device) * 
                (self.noise_scale / math.sqrt(self.n_coeffs))
            )

            self.linear_weights = nn.Parameter(
                torch.ones(input_dim, dtype=torch.float32, device=device) * 0.5
            )

            fine_n_coeffs = len(self.fine_control_points)
            self.fine_weights = nn.Parameter(
                torch.randn(input_dim, fine_n_coeffs, dtype=torch.float32, device=device) *
                (self.noise_scale * 0.3 / math.sqrt(fine_n_coeffs))
            )
            
            coarse_n_coeffs = len(self.coarse_control_points)
            self.coarse_weights = nn.Parameter(
                torch.randn(input_dim, coarse_n_coeffs, dtype=torch.float32, device=device) *
                (self.noise_scale * 0.8 / math.sqrt(coarse_n_coeffs))
            )
            
            self.attention_weights = nn.Parameter(
                torch.ones(input_dim, dtype=torch.float32, device=device)
            )

            self.main_res_weight = nn.Parameter(
                torch.rand(input_dim, dtype=torch.float32, device=device) + 0.5
            )
            self.fine_res_weight = nn.Parameter(
                torch.rand(input_dim, dtype=torch.float32, device=device) * 0.5
            )
            self.coarse_res_weight = nn.Parameter(
                torch.rand(input_dim, dtype=torch.float32, device=device) * 0.5
            )

            self.dynamic_mask = DynamicSparseMask(
                shape=(input_dim, self.n_coeffs),
                initial_sparsity=self.initial_sparsity,
                device=device
            )

    def _update_adaptive_grid(self, x: torch.Tensor):
        if not self.training:
            return
            
        with torch.no_grad():
            current_min = x.min().item()
            current_max = x.max().item()
            
            if self.update_count == 0:
                self.input_stats[0] = current_min
                self.input_stats[1] = current_max
            else:
                alpha = 0.01
                self.input_stats[0] = (1 - alpha) * self.input_stats[0] + alpha * current_min
                self.input_stats[1] = (1 - alpha) * self.input_stats[1] + alpha * current_max
            
            self.update_count += 1
            
            if self.update_count % 1000 == 0:
                margin = (self.input_stats[1] - self.input_stats[0]) * 0.15
                new_min = self.input_stats[0] - margin
                new_max = self.input_stats[1] + margin
                
                if abs(new_min - self.grid_min) > 0.5 or abs(new_max - self.grid_max) > 0.5:
                    self.grid_min.copy_(torch.tensor(new_min))
                    self.grid_max.copy_(torch.tensor(new_max))
                    self._create_control_points()

    def get_current_target_sparsity(self, total_steps: int) -> float:
        if total_steps == 0:
            return self.initial_sparsity

        progress = min(1.0, self.training_step.item() / total_steps)

        if self.dst_schedule == 'cosine':
            current_sparsity = self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * \
                               (1 - math.cos(math.pi * progress)) / 2
        elif self.dst_schedule == 'linear':
            current_sparsity = self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * progress
        elif self.dst_schedule == 'exponential':
            current_sparsity = self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * \
                               (1 - math.exp(-3 * progress))
        else:
            current_sparsity = self.target_sparsity

        return current_sparsity

    def dynamic_sparse_update(self, total_steps: int, update_frequency: int = 100):
        if self.dynamic_mask is None or not self.training:
            return

        self.training_step += 1

        if self.training_step % update_frequency != 0:
            return

        if self.spline_weights.grad is not None:
            self.dynamic_mask.update_importance_scores(
                self.spline_weights,
                self.spline_weights.grad
            )

        target_sparsity = self.get_current_target_sparsity(total_steps)
        current_sparsity = self.dynamic_mask.get_sparsity()

        if current_sparsity < target_sparsity:
            prune_ratio = min(0.1, (target_sparsity - current_sparsity) * 2)
            self.dynamic_mask.prune_connections(prune_ratio, strategy='magnitude')
        elif current_sparsity > target_sparsity * 1.1:
            grow_ratio = min(0.05, (current_sparsity - target_sparsity))
            self.dynamic_mask.grow_connections(grow_ratio, strategy='gradient', weights=self.spline_weights)

    def _enhanced_bspline_basis(self, x: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:

        distances_sq = (x.unsqueeze(-1) - control_points) ** 2

        sigma_adaptive = self.sigma * (1 + 0.1 * torch.sin(self.training_step.float() * 0.01) if self.training else 1)
        inv_sigma_sq_adaptive = 1.0 / (2 * sigma_adaptive ** 2)
        
        basis_values = torch.exp(-distances_sq * inv_sigma_sq_adaptive)

        row_sums = basis_values.sum(dim=-1, keepdim=True)
        basis_values = basis_values / (row_sums + 1e-8)
        
        return basis_values

    def get_visualization_data(self) -> Dict:
        """Extract data for visualization"""
        data = {
            'control_points': self.control_points.detach().cpu().numpy(),
            'fine_control_points': self.fine_control_points.detach().cpu().numpy(),
            'coarse_control_points': self.coarse_control_points.detach().cpu().numpy(),
            'spline_weights': self.spline_weights.detach().cpu().numpy() if self.spline_weights is not None else None,
            'fine_weights': self.fine_weights.detach().cpu().numpy() if self.fine_weights is not None else None,
            'coarse_weights': self.coarse_weights.detach().cpu().numpy() if self.coarse_weights is not None else None,
            'grid_min': self.grid_min.item(),
            'grid_max': self.grid_max.item(),
            'n_coeffs': self.n_coeffs,
            'input_dim': self.input_dim
        }
        

        if self.main_res_weight.ndim == 0:
             data['main_res_weight'] = self.main_res_weight.item()
             data['fine_res_weight'] = self.fine_res_weight.item()
             data['coarse_res_weight'] = self.coarse_res_weight.item()
        else:
             data['main_res_weight'] = self.main_res_weight.detach().cpu().numpy()
             data['fine_res_weight'] = self.fine_res_weight.detach().cpu().numpy()

        if self.attention_weights is not None:
            data['attention_weights'] = self.attention_weights.detach().cpu().numpy()
        else:
            data['attention_weights'] = None

        return data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

        if len(original_shape) == 2:
            batch_size, actual_input_dim = original_shape
            x_flat = x
        else:
            batch_size = original_shape[0]
            actual_input_dim = x[0].numel()
            x_flat = x.reshape(batch_size, actual_input_dim)

        self._initialize_weights(actual_input_dim, x.device)
        self._update_adaptive_grid(x_flat)

        attention_scores = torch.sigmoid(self.attention_weights)
        x_attended = x_flat * attention_scores.unsqueeze(0)

        grid_range = self.grid_max - self.grid_min
        x_clamped = torch.clamp(x_attended, 
                               self.grid_min - 0.2 * grid_range, 
                               self.grid_max + 0.2 * grid_range)

        if self.dynamic_mask is not None:
            masked_weights = self.dynamic_mask.apply_mask(self.spline_weights)
        else:
            masked_weights = self.spline_weights

        x_clamped_flat = x_clamped.contiguous().reshape(-1)
        basis_flat = self._enhanced_bspline_basis(x_clamped_flat, self.control_points)
        basis_reshaped = basis_flat.reshape(batch_size, actual_input_dim, self.n_coeffs)
        
        main_spline_output = torch.sum(basis_reshaped * masked_weights.unsqueeze(0), dim=-1)
        
        fine_basis_flat = self._enhanced_bspline_basis(x_clamped_flat, self.fine_control_points)
        fine_basis_reshaped = fine_basis_flat.reshape(batch_size, actual_input_dim, len(self.fine_control_points))
        fine_output = torch.sum(fine_basis_reshaped * self.fine_weights.unsqueeze(0), dim=-1)
        
        coarse_basis_flat = self._enhanced_bspline_basis(x_clamped_flat, self.coarse_control_points)
        coarse_basis_reshaped = coarse_basis_flat.reshape(batch_size, actual_input_dim, len(self.coarse_control_points))
        coarse_output = torch.sum(coarse_basis_reshaped * self.coarse_weights.unsqueeze(0), dim=-1)
        
        spline_output = (self.main_res_weight.unsqueeze(0) * main_spline_output + 
                         self.fine_res_weight.unsqueeze(0) * fine_output + 
                         self.coarse_res_weight.unsqueeze(0) * coarse_output)
        linear_output = x_flat * self.linear_weights.unsqueeze(0)
        
        output = linear_output + spline_output + 0.1 * x_flat

        return output.reshape(original_shape)


class KANLinear(nn.Module):
    DEFAULT_CONFIG = {
        'grid_size': 20,  
        'spline_order': 6,
        'grid_min': -3.0, 
        'grid_max': 3.0,
        'noise_scale': 0.1,
        
        'initial_sparsity': 0.0,
        'target_sparsity': 0.1,
        'dst_schedule': 'exponential',
  
        'enable_bias': True,
        'use_base_linear': True,  
        'base_activation': 'gelu',
        'mixing_weight_init': 0.9,  
    }

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        config = self.DEFAULT_CONFIG.copy()
        config['enable_bias'] = bias
        if config['use_base_linear']:
            self.base_linear = nn.Linear(in_features, out_features, bias=config['enable_bias'])
        else:
            self.base_linear = None

        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(), 
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            None: None
        }
        self.base_activation = activation_map.get(config['base_activation'])

        self.spline_transform = BSplineActivation(
            grid_size=config['grid_size'],
            spline_order=config['spline_order'],
            grid_min=config['grid_min'],
            grid_max=config['grid_max'],
            noise_scale=config['noise_scale'],
            initial_sparsity=config['initial_sparsity'],
            target_sparsity=config['target_sparsity'],
            dst_schedule=config['dst_schedule']
        )

        self.mixing_weight = nn.Parameter(torch.tensor(config['mixing_weight_init'], dtype=torch.float32))
        

        self.spline_output_projection = nn.Linear(in_features, out_features, bias=False)
        if in_features <= 20: 
            self.interaction_weights = nn.Parameter(
                torch.randn(in_features, in_features, out_features) * 0.01
            )
        else:
            self.interaction_weights = None
        
        self.layer_norm = nn.LayerNorm(in_features)
        
        self._reset_parameters()

    def _reset_parameters(self):
        if self.base_linear is not None:
            nn.init.kaiming_uniform_(self.base_linear.weight, a=math.sqrt(5))
            if self.base_linear.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_linear.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.base_linear.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.spline_output_projection.weight, a=math.sqrt(5))

    def _compute_interactions(self, x: torch.Tensor) -> torch.Tensor:
        if self.interaction_weights is None:
            return torch.zeros(x.shape[0], self.out_features, device=x.device)
        
        batch_size = x.shape[0]
        interactions = torch.zeros(batch_size, self.out_features, device=x.device)
        
        n_interactions = min(self.in_features * (self.in_features - 1) // 4, 50)
        indices = torch.randperm(self.in_features * self.in_features)[:n_interactions]
        
        for idx in indices:
            i = idx // self.in_features
            j = idx % self.in_features
            if i != j:
                interaction_term = x[:, i:i+1] * x[:, j:j+1]
                weighted_interaction = interaction_term * self.interaction_weights[i, j].unsqueeze(0)
                interactions += weighted_interaction * 0.1 
        
        return interactions

    def dynamic_sparse_update(self, total_steps: int, update_frequency: int = 100):
        self.spline_transform.dynamic_sparse_update(total_steps, update_frequency)

    def get_sparsity_stats(self) -> Dict:
        if hasattr(self.spline_transform, 'dynamic_mask') and self.spline_transform.dynamic_mask:
            sparsity = self.spline_transform.dynamic_mask.get_sparsity()
        else:
            sparsity = 0.0
            
        stats = {
            'layer_type': 'KANLinear',
            'in_features': self.in_features,
            'out_features': self.out_features,
            'sparsity': sparsity,
            'has_interactions': self.interaction_weights is not None,
            'grid_size': self.spline_transform.grid_size,
        }
        return stats

    def get_visualization_data(self) -> Dict:
        """Extract data for visualization"""
        data = {
            'layer_type': 'KANLinear',
            'in_features': self.in_features,
            'out_features': self.out_features,
            'spline_transform_data': self.spline_transform.get_visualization_data(),
            'spline_output_projection': self.spline_output_projection.weight.detach().cpu().numpy() if self.spline_output_projection is not None else None,
            'base_linear': self.base_linear.weight.detach().cpu().numpy() if self.base_linear is not None else None,
            'mixing_weight': self.mixing_weight.detach().cpu().numpy()
        }
        return data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input features {self.in_features}, got {x.shape[-1]}")

        original_shape = x.shape
        needs_reshape = len(original_shape) > 2
        
        if needs_reshape:
            batch_dims = original_shape[:-1]
            x_flat = x.reshape(-1, self.in_features)
        else:
            x_flat = x


        x_normalized = self.layer_norm(x_flat)

        linear_out = None
        if self.base_linear is not None:
            linear_out = self.base_linear(x_normalized)
            if self.base_activation is not None:
                linear_out = self.base_activation(linear_out)

        if self.base_activation is not None and self.base_linear is None:
            spline_input = self.base_activation(x_normalized)
        else:
            spline_input = x_normalized
            
        spline_raw_output = self.spline_transform(spline_input)
        spline_out = self.spline_output_projection(spline_raw_output)

        interaction_out = self._compute_interactions(x_normalized)

        if self.base_linear is not None:
            mix_weight = torch.sigmoid(self.mixing_weight)
            output = (1 - mix_weight) * linear_out + mix_weight * spline_out + interaction_out
        else:
            output = spline_out + interaction_out

        if needs_reshape:
            output = output.view(*batch_dims, self.out_features)

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.base_linear is not None and self.base_linear.bias is not None}'