import math
from sympy import true
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.layers import CALayer, SHALayer, SHALayerV2, CALayerV2
from scripts.utils import pad_tensor, pad_tensor_back
import pdb
import numbers
from torch import Tensor
from einops import rearrange
from torch.fft import fftshift

def exists(x):
    return x is not None

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1) 
        return x

class TemporalAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False): 
        super(TemporalAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed, temporal_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1) #每个通道都平行加上noise
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 2, 1)
    def forward(self, x, y):
        return self.conv(x),self.conv2(y)

class NoiseFusedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
                noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x,time_emb=None):
        b, c, h, w = x.shape
        h = self.block1(x)
        if time_emb is not None:
            h = self.noise_func(h, time_emb)
        return h + self.res_conv(x)

class FirstConv(nn.Module):
    def __init__(self, in_channel,inner_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,inner_channel,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channel,inner_channel,kernel_size=3,padding=1)
    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        return x,y


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

from dataclasses import dataclass
@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation_old(nn.Module):
    def __init__(self, cond_dim: int, dim:int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 9 if double else 3
        self.lin = nn.Linear(cond_dim, self.multiplier * dim, bias=True)  

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        batch = vec.shape[0]

        out = self.lin(nn.functional.silu(vec))[:, None, :].view(
                batch, -1, 1, 1).chunk(self.multiplier, dim=1) 
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class Modulation(nn.Module):
    def __init__(self, cond_dim: int, dim:int):
        super().__init__()
        self.multiplier = 3
        self.lin = nn.Linear(cond_dim, self.multiplier * dim, bias=True) 

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        batch = vec.shape[0]

        out = self.lin(nn.functional.silu(vec))[:, None, :].view(
                batch, -1, 1, 1).chunk(self.multiplier, dim=1) 
        return (
            ModulationOut(*out)
        )

def qkv_split(tensor,num_heads):
    q,k,v = tensor.chunk(3, dim=1)        
    q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=num_heads)
    k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=num_heads)
    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=num_heads)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    return q,k,v

class DualDomain_prompter(nn.Module):
    def __init__(self,in_dim,prompt_dim,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(DualDomain_prompter,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.conv3x3_rgb = nn.Conv2d(in_dim,64,kernel_size=3,stride=1,bias=False)
        self.conv3x3_fft = nn.Conv2d(in_dim,64,kernel_size=3,stride=1,bias=False)
        
        self.prompt_len = prompt_len
        self.linear_layer = nn.Linear(64*2, self.prompt_len)
    def forward(self,x):
        B,C,H,W = x.shape
        fft = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft)
        fft = torch.abs(fft_shifted)
        embx = self.conv3x3_rgb(x)
        emby = self.conv3x3_fft(fft)
        
        emb = torch.cat([embx,emby],dim=1).mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        return prompt
      
      
class DualStreamResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, ffn_factor=2.66,num_heads=4,\
        with_attn=False, use_affine_level=False, attn_type=None):
        super().__init__()
        self.img1_mod1 = Modulation(noise_level_emb_dim, dim)
        self.img2_mod1 = Modulation(noise_level_emb_dim, dim)
        self.img1_mod2 = Modulation(noise_level_emb_dim, dim_out)
        self.img2_mod2 = Modulation(noise_level_emb_dim, dim_out)
        self.img1_norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.img1_norm2 = LayerNorm(dim_out, LayerNorm_type='WithBias')
        self.img2_norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.img2_norm2 = LayerNorm(dim_out, LayerNorm_type='WithBias')
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkvconv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkvconv2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.num_heads = num_heads
        self.ffn1_1 = FeedForward(dim, ffn_factor, bias=False)
        self.ffn2_1 = FeedForward(dim, ffn_factor, bias=False)
        self.res_block1 = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, use_affine_level=use_affine_level)
        self.res_block2 = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, use_affine_level=use_affine_level)
        
        self.ffn1_2 = FeedForward(dim_out, ffn_factor, bias=False)
        self.ffn2_2 = FeedForward(dim_out, ffn_factor, bias=False)
        

    def forward(self, x, y, time_emb, prompt):
        _,c,h,w = x.shape
           
        x = self.img1_norm1(x)
        y = self.img2_norm1(y)
        
        img1_mod1 = self.img1_mod1(prompt)
        x = (1 + img1_mod1.scale) * x + img1_mod1.shift
        img2_mod1 = self.img2_mod1(time_emb)
        y = (1 + img2_mod1.scale) * y + img2_mod1.shift
        x = self.qkvconv1(x)
        y = self.qkvconv2(y)
        img1_q,img1_k,img1_v = qkv_split(x,num_heads=self.num_heads)
        img2_q,img2_k,img2_v = qkv_split(y,num_heads=self.num_heads)
        q = torch.cat((img1_q, img2_q), dim=2)
        k = torch.cat((img1_k, img2_k), dim=2)
        v = torch.cat((img1_v, img2_v), dim=2)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x,y = out.chunk(2,dim=1)
        x = x + img1_mod1.gate * self.ffn1_1(x)
        y = y + img2_mod1.gate * self.ffn2_1(y)
        x = self.res_block1(x,prompt)
        y = self.res_block2(y,time_emb)

        img1_mod2 = self.img1_mod2(prompt)
        img2_mod2 = self.img2_mod2(time_emb)
        
        x = x + img1_mod2.gate * self.ffn1_2((1 + img1_mod2.scale) * self.img1_norm2(x) + img1_mod2.shift)
        y = y + img2_mod2.gate * self.ffn2_2((1 + img2_mod2.scale) * self.img2_norm2(y) + img2_mod2.shift)
        return x, y


class SingleResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, ffn_factor=2.66,\
        with_attn=False, use_affine_level=False, attn_type=None):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
                noise_level_emb_dim, dim_out, use_affine_level)
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, use_affine_level=use_affine_level)
        self.ffn = FeedForward(dim_out, ffn_factor, bias=False)
        self.norm = LayerNorm(dim_out, LayerNorm_type='WithBias')


    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        h = self.noise_func(x, time_emb)
        x = x + self.ffn(self.norm(h))

        return x 
    

class Rectified_fusion_block(nn.Module):
    def __init__(self, dim, dim_res ,dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, ffn_factor=2.66,num_heads=4,\
        with_attn=False, use_affine_level=False, attn_type=None):
        super().__init__()
        self.img1_mod1 = Modulation(noise_level_emb_dim, dim)
        self.img3_mod1 = Modulation(noise_level_emb_dim, dim_res)
        self.img_mod2 = Modulation(noise_level_emb_dim, dim_out)
        self.img1_norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.img3_norm1 = LayerNorm(dim_res, LayerNorm_type='WithBias')
        self.img_norm2 = LayerNorm(dim_out, LayerNorm_type='WithBias')
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qvconv = nn.Conv2d(dim, dim*2 , kernel_size=1, bias=False)
        self.kconv = nn.Conv2d(dim_res, dim , kernel_size=1, bias=False)
        self.num_heads = num_heads

        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, use_affine_level=use_affine_level)
        self.ffn = FeedForward(dim_out, ffn_factor, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.gelu = nn.GELU()
        self.w = nn.Parameter(torch.ones(2))
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, y, time_emb):
        _,c,h,w = x.shape

        x = self.img1_norm1(x)
        y = self.img3_norm1(y)

        img1_mod1 = self.img1_mod1(time_emb)
        x = (1 + img1_mod1.scale) * x + img1_mod1.shift
        img3_mod1 = self.img3_mod1(time_emb)
        y = (1 + img3_mod1.scale) * y + img3_mod1.shift
        
        q,k = self.qvconv(x).chunk(2,dim=1)
        v = self.kconv(y)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        attn0 = self.softmax(attn)
        attn1 = self.gelu(attn)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))  
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))  
        attn = attn0 * w1 + attn1 * w2  

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x = x + out
    
        x = self.res_block(x,time_emb)
        img_mod2 = self.img_mod2(time_emb)
        x = x + img_mod2.gate * self.ffn((1 + img_mod2.scale) * self.img_norm2(x) + img_mod2.shift)

        return x
class Expert(nn.Module):
    def __init__(self, n_embd,dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear =nn.Linear(n_embed, num_experts)
    
    def forward(self, mh_output):
        logits = self.linear(mh_output)   
        
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        
        zeros = torch.full_like(logits, float('-inf'))
        
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)
    
    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class ShareMLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.hidden_size = n_embed
        self.gate_proj = nn.Linear(self.hidden_size, 4*self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, 4*self.hidden_size, bias=False)
        self.down_proj = nn.Linear(4*self.hidden_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class Task_adaptive_embedding(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(Task_adaptive_embedding, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.num_share = 2
        self.share_experts = ShareMLP(n_embed)
    def forward(self, x):
        
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)
       
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            
            expert_mask = (indices == i).any(dim=-1)
            
            flat_mask = expert_mask.view(-1)
           
            if flat_mask.any():
                
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] += weighted_output.squeeze(1)
        
        share_output = self.share_experts(x)
        final_output = final_output + share_output
        return final_output


@ARCH_REGISTRY.register()
class DualUNet(nn.Module):
    def __init__(
        self,
        in_channel=13,
        out_channel=3,
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 4),
        attn_res=(16),
        num_heads=(4,6,8),
        res_blocks=2,
        dropout=0.2,
        with_noise_level_emb=True,
        image_size=256,
        ffn_factor=2.66,
        use_affine_level=False,
        attn_type=None,
        divide=None,
        drop2d_input=False,
        drop2d_input_p=0.0,
        channel_randperm_input=False
    ):
        super().__init__()
        self.drop2d_input = drop2d_input
        if self.drop2d_input:
            self.drop2d_in = nn.Dropout2d(drop2d_input_p)
        
        self.channel_randperm_input = channel_randperm_input

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        
        self.divide = divide

        num_mults = len(channel_mults)  
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = []
        downs.append(FirstConv(in_channel,inner_channel))
        for ind in range(num_mults): 
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(DualStreamResnetBlocWithAttn( 
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn,\
                    ffn_factor=ffn_factor,num_heads=num_heads[ind], use_affine_level=use_affine_level, \
                        attn_type=attn_type))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))  
                feat_channels.append(pre_channel)
                now_res = now_res//2  
        self.downs = nn.ModuleList(downs)
        mid_use_attn =(len(attn_res) != 0)
        
        self.mid = nn.ModuleList([
            DualStreamResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,ffn_factor=ffn_factor,
                               num_heads=num_heads[-1],dropout=dropout, with_attn=mid_use_attn, use_affine_level=use_affine_level,
                               attn_type=attn_type),
            Rectified_fusion_block(pre_channel, pre_channel ,pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=0, ffn_factor=ffn_factor,num_heads=4,\
                with_attn=False, use_affine_level=use_affine_level, attn_type=attn_type)
            
        ]) 
        
        ups = [] 
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(
                    SingleResnetBlocWithAttn(  
                    pre_channel+2*(feat_channels.pop()), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,ffn_factor=ffn_factor,
                        dropout=dropout, with_attn=use_attn, use_affine_level=use_affine_level, attn_type=attn_type)
                    )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel)) 
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        self.DDP = DualDomain_prompter(in_dim=3,prompt_dim=inner_channel,prompt_len=32,prompt_size = in_channel,lin_dim = in_channel)
        self.AP = nn.AdaptiveAvgPool2d(1)
        self.TAE = Task_adaptive_embedding(n_embed=inner_channel, num_experts=4, top_k=2)


    def forward(self, x, y, time=None):
        if self.channel_randperm_input:
            from scripts.pytorch_utils import channel_randperm
            x[:, :6, ...] = channel_randperm(x[:, :6, ...])
            y[:, :6, ...] = channel_randperm(y[:, :6, ...])
        if self.drop2d_input:
            x[:, :6, ...] = self.drop2d_in(x[:, :6, ...])
            y[:, :6, ...] = self.drop2d_in(y[:, :6, ...])
        if self.divide:
            x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x, self.divide) 
            y, _, _, _, _ = pad_tensor(y, self.divide)
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None  
        b,_,_,_=x.shape
  
        fft = x
        prompt = self.DDP(fft)
        prompt = self.AP(prompt).squeeze(-1).transpose(2,1)
        t = t + self.TAE(prompt)
        feats_x,feats_y= [],[]
        for layer in self.downs:
            if isinstance(layer, DualStreamResnetBlocWithAttn):
                x,y = layer(x, y, t,prompt)
            else:
                x,y = layer(x,y)
            feats_x.append(x)
            feats_y.append(y)

        for layer in self.mid:
            if isinstance(layer, DualStreamResnetBlocWithAttn):
                x,y = layer(x, y, t,prompt)
            elif isinstance(layer, Rectified_fusion_block):
                t = t + prompt
                x = layer(x, y, t)
 
            else:
                x = layer(x)
        
        for layer in self.ups:
            if isinstance(layer, SingleResnetBlocWithAttn):
                x = torch.cat((x, feats_x.pop(), feats_y.pop()), dim=1)
                x = layer(x, t) 
            else:
                x = layer(x)
        if self.divide:
            out = self.final_conv(x)
            out = pad_tensor_back(out, pad_left, pad_right, pad_top, pad_bottom)
            return out
        else:
            return self.final_conv(x)


