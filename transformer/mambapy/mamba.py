import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

# Import pscan with fallback for cloud environment
try:
    from transformer.mambapy.pscan import pscan
except ImportError:
    # Fallback: disable pscan if import fails
    pscan = None

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)
    
    # Dataset-specific optimization flag
    dataset_type: str = 'default' # 'so', 'retweet', or 'default'
    
    # Ablation study flags
    ablation_mode: str = 'full' # 'full', 'time_scaling_only', 'dual_channel_only'

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization (Following Original TV-MHP exactly)
        A = repeat(torch.arange(1, config.d_state + 1), 'n -> d n', d=config.d_inner)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y) # (B, L, D)
            return output

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm(self, x, z):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)
        
        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2) # (B, L, ED)
        
        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h
    
class Mambadelta(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlockdelta(config) for _ in range(config.n_layers)])

    def forward(self, x, delta):
        # x : (B, L, D)

        # y : (B, L, D)

        latest_scaling_factor = None
        for layer in self.layers:
            x, latest_scaling_factor = layer(x, delta)

        return x, latest_scaling_factor
    """
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches
    """
class ResidualBlockdelta(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlockdelta(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        
        # Gated residual connection for efficiency
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, delta):
        # x : (B, L, D)

        # output : (B, L, D)
        # Pre-norm ResNet structure with gated residual connection
        normed_input = self.norm(x)
        mixer_output, latest_scaling_factor = self.mixer(normed_input, delta)

        # Gated residual connection with learnable weight
        output = x + self.gate * mixer_output
        return output, latest_scaling_factor
    """
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache
    """
class MambaBlockdelta(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # GRU Time-Scaling Mechanism
        # Maintain input-output dimension consistency
        self.gru = nn.GRU(config.d_inner, config.d_inner, batch_first=True)
        # Scaling factor: s(L_j) = Softplus(W_l^T l_j + b_l)
        self.W_T_l = nn.Linear(config.d_inner, 1, bias=True)
        self.softplus = nn.Softplus()
        
        # Dual-Channel State Transition: Event 2/3, Time 1/3
        self.event_dim = (2 * config.d_state) // 3  # Event channel: 2/3 of d_state
        self.time_dim = config.d_state - self.event_dim  # Time channel: 1/3 of d_state
        self.W_c = nn.Linear(config.d_inner, self.event_dim, bias=False)
        self.W_tau = nn.Linear(1, self.time_dim, bias=False)
        self.B_0 = nn.Parameter(torch.ones(config.d_state))
        
        # projects x to input-dependent delta, B, C (Following Original TV-MHP)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)

        # projects delta from dt_rank to d_inner (Following Original TV-MHP)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization (Following Original TV-MHP)
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # delta bias initialization (Following Original TV-MHP)
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization (Following Original TV-MHP exactly)
        A = repeat(torch.arange(1, config.d_state + 1), 'n -> d n', d=config.d_inner)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x, delta):
        # x : (B, L, D)
        
        # y : (B, L, D)

        _, L, _ = x.shape

        # First apply in_proj to get d_inner features
        xz = self.in_proj(x) # (B, L, d_model) → (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1) # (B, L, d_inner), (B, L, d_inner)

        # x branch - Convolution with optimized boundary handling  
        x_conv = x_inner.transpose(1, 2) # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L] # depthwise convolution over time, with a short filter
        x_conv = x_conv.transpose(1, 2) # (B, L, d_inner)
        
        # Apply SiLU activation
        x_conv = F.silu(x_conv)
        
        # Ablation study logic
        if self.config.ablation_mode == 'time_scaling_only':
            # Only Time-Scaling mechanism, no dual-channel
            h_gru, _ = self.gru(x_conv)
            scaling_raw = self.W_T_l(h_gru)
            scaling_raw = torch.clamp(scaling_raw, min=-10.0, max=10.0)
            scaling_factors = self.softplus(scaling_raw)
            scaling_factors = torch.clamp(scaling_factors, min=0.1, max=10.0)
            time_gaps_expanded = delta.unsqueeze(-1)
            scaled_time_gaps = time_gaps_expanded * scaling_factors
        elif self.config.ablation_mode == 'dual_channel_only':
            # Only Dual-Channel mechanism, no time-scaling
            scaling_factors = torch.ones_like(delta.unsqueeze(-1))  # No scaling
            time_gaps_expanded = delta.unsqueeze(-1)
            scaled_time_gaps = time_gaps_expanded  # Use original time intervals
        else:
            # Full model with both mechanisms
            h_gru, _ = self.gru(x_conv)  # (B, L, d_inner) - fully parallel
            
            # Compute scaling factor: s(L_j) = Softplus(W_l^T l_j + b_l)
            scaling_raw = self.W_T_l(h_gru)  # (B, L, 1)
            # Gradient explosion protection: limit scaling factor input range
            if self.config.dataset_type == 'so':
                # SO dataset: relaxed scaling for better accuracy
                scaling_raw = torch.clamp(scaling_raw, min=-15.0, max=15.0)  # Wider range
                scaling_factors = self.softplus(scaling_raw)
                scaling_factors = torch.clamp(scaling_factors, min=0.05, max=15.0)  # More flexible
            elif self.config.dataset_type == 'retweet':
                # Retweet dataset: relaxed scaling for LL recovery
                scaling_raw = torch.clamp(scaling_raw, min=-12.0, max=12.0)  # Wider range
                scaling_factors = self.softplus(scaling_raw)
                scaling_factors = torch.clamp(scaling_factors, min=0.08, max=12.0)  # More flexible
            else:
                # Default clamp for other datasets
                scaling_raw = torch.clamp(scaling_raw, min=-10.0, max=10.0)
                scaling_factors = self.softplus(scaling_raw)
                scaling_factors = torch.clamp(scaling_factors, min=0.1, max=10.0)
            
            # Compute scaled time intervals
            time_gaps_expanded = delta.unsqueeze(-1)  # (B, L, 1)
            scaled_time_gaps = time_gaps_expanded * scaling_factors  # (B, L, 1)
        
        # Use scaled delta in SSM
        y, latest_scaling_factor = self.ssm(x_conv, z, scaled_time_gaps, scaling_factors)

        # Output projection (z gate already applied in ssm)
        output = self.out_proj(y) # (B, L, D)
        return output, latest_scaling_factor
    
    def ssm(self, x_conv, z, scaled_time_gaps, scaling_factors):
        # x_conv : (B, L, d_inner) - 已经过conv和SiLU处理
        # scaled_time_gaps : (B, L, 1)

        # y : (B, L, d_inner)
        _, L, d_inner = x_conv.shape

        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        D = self.D.float()

        # Ablation-aware dual-channel mechanism
        if self.config.ablation_mode == 'time_scaling_only':
            # No dual-channel, use standard x_proj for B and C calculation
            x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
            _, B_dual, C = x_dbl.split([self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
            B_dual = torch.sigmoid(B_dual)  # Standard B processing
            # Get base_dt for consistency
            base_dt, _ = self.compute_ssm_params(x_conv)
        else:
            # Use dual-channel mechanism (both 'dual_channel_only' and 'full' modes)
            content_channel = self.W_c(x_conv)  # (B, L, d_inner) → (B, L, event_dim=10)
            temporal_channel = self.W_tau(scaled_time_gaps)  # (B, L, 1) → (B, L, time_dim=6)
            concat_channels = torch.cat([content_channel, temporal_channel], dim=-1)  # (B, L, d_state)
            B_dual = torch.sigmoid(concat_channels) * self.B_0.unsqueeze(0).unsqueeze(0)  # (B, L, d_state)
            # Compute C separately for dual-channel modes
            base_dt, C = self.compute_ssm_params(x_conv)
        
        # Time-scaling logic: Δ'_i = Δ_i × s(L_i), scaled interval used throughout SSM
        
        # Convert scaled time intervals to differential form
        scaled_diff = torch.diff(scaled_time_gaps.squeeze(-1), dim=1)  # (B, L-1)
        scaled_first = scaled_time_gaps[:, :1, 0]  # (B, 1)
        scaled_intervals = torch.cat([scaled_first, scaled_diff], dim=1)  # (B, L)
        
        # Convert to SSM dt format: use scaled time intervals directly
        dt_scaled = scaled_intervals.unsqueeze(-1).expand(-1, -1, d_inner)  # (B, L, d_inner)
        
        # Combine with neural adaptive modulation
        dt_scaled = dt_scaled * base_dt
        
        # Gradient explosion protection: limit delta range
        if self.config.dataset_type == 'so':
            # SO dataset: wider dt range for better time modeling
            dt_scaled = torch.clamp(dt_scaled, min=5e-7, max=15.0)
        elif self.config.dataset_type == 'retweet':
            # Retweet dataset: wider dt range for LL recovery  
            dt_scaled = torch.clamp(dt_scaled, min=8e-7, max=12.0)
        else:
            # Default clamp for other datasets
            dt_scaled = torch.clamp(dt_scaled, min=1e-6, max=10.0)
        
        # Protection for deltaA computation to avoid exponential explosion
        delta_A_product = einsum(dt_scaled, A, 'b l d, d n -> b l d n')
        if self.config.dataset_type == 'so':
            # SO dataset: wider deltaA range for better dynamics
            delta_A_product = torch.clamp(delta_A_product, min=-25.0, max=25.0)
        elif self.config.dataset_type == 'retweet':
            # Retweet dataset: wider deltaA range for LL recovery
            delta_A_product = torch.clamp(delta_A_product, min=-22.0, max=22.0)
        else:
            # Default clamp for other datasets
            delta_A_product = torch.clamp(delta_A_product, min=-20.0, max=20.0)
        deltaA = torch.exp(delta_A_product)  # (B, L, d_inner, d_state)
        
        deltaB_u = einsum(dt_scaled, B_dual, x_conv, 'b l d, b l n, b l d -> b l d n')
        
        # Efficient parallel scan optimized for TPP sequences
        if self.config.pscan and pscan is not None:
            # Use optimized parallel scan for long sequence efficiency
            x_state = pscan(deltaA, deltaB_u)
            y = einsum(x_state, C, 'b l d n, b l n -> b l d')
        else:
            # Sequential scan as fallback
            x_state = deltaB_u.new_zeros((x_conv.size(0), d_inner, self.config.d_state))
            ys = []
            for i in range(L):
                x_state = deltaA[:, i] * x_state + deltaB_u[:, i]
                y_i = einsum(x_state, C[:, i], 'b d n, b n -> b d')
                ys.append(y_i)
            y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        
        # Add skip connection
        y = y + x_conv * D.unsqueeze(0).unsqueeze(0)
        
        # Apply z gate control
        y = y * F.silu(z)

        # Return the latest scaling factor for time prediction adjustment
        latest_scaling_factor = scaling_factors[:, -1, :]  # (B, 1)
        
        return y, latest_scaling_factor
    
    def compute_ssm_params(self, x):
        """Compute SSM parameters except B (handled by dual-channel mechanism)"""
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, _, C = x_dbl.split([self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        # B is computed separately by dual-channel mechanism
        dt = self.dt_proj(dt)  # (B, L, d_inner) - complete dt computation
        dt = F.softplus(dt + self.dt_proj.bias)
        return dt, C
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    """
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h
        """

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    