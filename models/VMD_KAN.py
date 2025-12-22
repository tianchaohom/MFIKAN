from torch import nn
import torch
from layers.VMDDecomp import VMDDecomp
from layers.KAN import KAN
from layers.KANlayers import KANLinear
from layers.Embed import FeatureFusionEmbedding
from layers.revin import RevIN

class Projector(nn.Module):
    def __init__(self, d_model, pred_len):
        super().__init__()
        self.proj = nn.Sequential(
            KANLinear(d_model, d_model * 2),
            KANLinear(d_model * 2, d_model),
            KANLinear(d_model, pred_len)
        )
    
    def forward(self, x):
        return self.proj(x)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.c_in = configs.c_in
        self.K = configs.vmd_scales
        self.revin = configs.revin
        self.revin_layer = RevIN(self.c_in, affine=True, subtract_last=False)
        self.use_vmd = True
        self.vmd_decomp = VMDDecomp(configs)
        
        self.embeddings = nn.ModuleList([
            FeatureFusionEmbedding(
                configs.seq_len,
                configs.d_model,
                getattr(configs, 'embed', 'timeF'),
                getattr(configs, 'freq', 'h'),
                getattr(configs, 'dropout', 0.1),
                configs=configs  
            )
            for _ in range(self.K)
        ])
        
        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([
                KAN(configs.d_model, configs.enc_in)
                for _ in range(configs.e_layers)
            ])
            for _ in range(self.K)
        ])

        self.projector = Projector(configs.d_model, configs.pred_len)
    
    def forward(self, x):
        B, L, M = x.shape
        
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        if self.use_vmd:
        
            imf_list = self.vmd_decomp(x)
            scale_outputs = []
            for k in range(self.K):
                x_k = imf_list[k].permute(0, 2, 1)
                x_k = self.embeddings[k](x_k, None)
                for mod in self.encoder_layers[k]:
                    x_k = mod(x_k)
                scale_outputs.append(x_k)
            x = torch.stack(scale_outputs, dim=-1).sum(dim=-1)
        else:
            x_k = x.permute(0, 2, 1)  # [B, M, L]
            x = self.embeddings[0](x_k, None)
            for mod in self.encoder_layers[0]:
                x = mod(x)
        
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1)
        
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
        
        return dec_out[:, -self.pred_len:, :]