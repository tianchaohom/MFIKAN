from torch import nn
from layers.KANlayers import  KANLinear

class KAN(nn.Module):
    """Encoder with KAN layers"""
    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            KANLinear(d_model, d_model),
        )

        self.ff2 = nn.Sequential(
            KANLinear(enc_in, enc_in)
        )

    def forward(self, x):
        y_feat = self.ff1(x)
        y_feat = x + y_feat
        y_feat = self.norm1(y_feat)
        y_temp = y_feat.permute(0, 2, 1)
        y_temp = self.ff2(y_temp)
        y_temp = y_temp.permute(0, 2, 1)
        output = x + y_feat + y_temp
        output = self.norm2(output)
        
        return output