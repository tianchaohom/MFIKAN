import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.KANlayers import KANLinear

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Modified to handle [B, M, L] input"""
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.linear = nn.Linear(c_in, d_model)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [B, M, L] -> [B, M, d_model]
        return self.linear(x)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class FeatureEmbedding(nn.Module):
    """
    [B,M,L] → [B,M,D] → [B,D,M] → [B,D,1] → [B,D,M] → [B,M,D]
    """
    def __init__(self, c_in, d_model, configs=None):
        super().__init__()
        self.c_in = c_in  
        self.d_model = d_model
        
        self.enc_in = getattr(configs, 'enc_in', c_in) if configs else c_in
        
        self.projection = nn.Linear(c_in, d_model)
        
        self.feature_aggregation = KANLinear(self.enc_in, 1)
        
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, x):
    
        B, M, L = x.shape
        
        x_proj = self.projection(x)  

        x_transposed = x_proj.transpose(1, 2)  

        x_aggregated = self.feature_aggregation(x_transposed) 
        
        x_expanded = x_aggregated.expand(-1, -1, M) 
        
        feature_emb = x_expanded.transpose(1, 2)  
        
        return feature_emb


class FeatureFusionEmbedding(nn.Module):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, configs=None):
        super(FeatureFusionEmbedding, self).__init__()
        
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq
            )
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq
            )
        
        self.feature_embedding = FeatureEmbedding(c_in=c_in, d_model=d_model, configs=configs)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, x_mark=None):
 
        # 1. Value embedding
        value_emb = self.value_embedding(x)  
        
        # 2. Position embedding
        pos_emb = self.position_embedding(value_emb)  
        
        # 3. Feature embedding 
        feat_emb = self.feature_embedding(x)  
        
        # 4. Temporal embedding 
        if x_mark is not None:
            temp_emb = self.temporal_embedding(x_mark)  
            combined_emb = value_emb + pos_emb + feat_emb + temp_emb
        else:
            combined_emb = value_emb + pos_emb + feat_emb
        
        # 5. Dropout
        return self.dropout(combined_emb)

