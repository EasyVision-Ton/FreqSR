import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np
import cv2, os


def normalizaion_gray(x):
    B, H, W = x.shape

    x = (x - x.min()) / (x.max() - x.min())
    return x


def print_intermediate_feature(x, save_path='./results/intermediate_features', name='', negetive=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        B, H, W, C = x.shape

        choosed_feat_0 = x[:, :H, :W, 0]
        choosed_feat_1 = x[:, :H, :W, C//2]
        choosed_feat_2 = x[:, :H, :W, -1]

        if negetive:  
            choosed_feat_0 = normalizaion_gray(choosed_feat_0)
            choosed_feat_1 = normalizaion_gray(choosed_feat_1)
            choosed_feat_2 = normalizaion_gray(choosed_feat_2)
            
        choosed_feat_0 = choosed_feat_0.squeeze(0).cpu().clamp(0, 1).numpy()
        choosed_feat_1 = choosed_feat_1.squeeze(0).cpu().clamp(0, 1).numpy()
        choosed_feat_2 = choosed_feat_2.squeeze(0).cpu().clamp(0, 1).numpy()

        choosed_feat_0 = np.round(choosed_feat_0 * 255.0).astype(np.uint8)
        choosed_feat_1 = np.round(choosed_feat_1 * 255.0).astype(np.uint8)
        choosed_feat_2 = np.round(choosed_feat_2 * 255.0).astype(np.uint8)

        c_f_0 = cv2.cvtColor(choosed_feat_0, cv2.COLOR_GRAY2RGB)[:, :, ::-1]
        c_f_1 = cv2.cvtColor(choosed_feat_1, cv2.COLOR_GRAY2RGB)[:, :, ::-1]
        c_f_2 = cv2.cvtColor(choosed_feat_2, cv2.COLOR_GRAY2RGB)[:, :, ::-1]

        c_f = {0: c_f_0,
               1: c_f_1,
               2: c_f_2}
        
        for i in range(3):
            addr = name + '_' + str(i) + '.png'
            addr = os.path.join(save_path, addr)
            cv2.imwrite(addr, c_f[i])


def print_single_feature(x, save_path='./results_intermediate', name='', negetive=False, index=0):
        B, H, W, C = x.shape

        index = int(index)

        choosed_feat = x[:, :H, :W, index]

        if negetive:  
            choosed_feat = normalizaion_gray(choosed_feat)
            
        choosed_feat = choosed_feat.squeeze(0).cpu().clamp(0, 1).numpy()

        choosed_feat = np.round(choosed_feat * 255.0).astype(np.uint8)

        c_f = cv2.cvtColor(choosed_feat, cv2.COLOR_GRAY2RGB)[:, :, ::-1]
        
       
        addr = name + '_' + str(index) + '.png'
        addr = os.path.join(save_path, addr)
        cv2.imwrite(addr, c_f)


def print_intermediate_features(x, save_path='./results/intermediate_features', name='', negetive=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        B, H, W, C = x.shape

        indexes = list(range(20))
        for i in indexes:
            print_single_feature(x, save_path, name, index=i)


class Mlp(nn.Module):
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


class FourierUnit_H(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho', if_sparse=True, groups=2, enhanced=True):
        # bn_layer not used
        super(FourierUnit_H, self).__init__()
        self.enhanced = enhanced

        # single FC
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, 
                                          embed_dim * 2, 
                                          1, 1, 0,
                                          groups=groups)
        
        # double FC
        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0, groups=groups),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0, groups=groups)
        # )
        
        if enhanced:
            self.conv_layer_ = nn.Conv2d(2, 1, 1, 1, 0)
        
        if if_sparse:
            
            # self.relu = nn.ReLU()
            # self.relu = nn.Softshrink(lambd=0.1)
            self.relu = nn.Sigmoid()
            
        else:    
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fft_norm = fft_norm

    def enhanced_feat(self, x):
        b, c, h, w = x.shape

        max_f = torch.max(x, dim=1, keepdim=True)[0]  # b 1 h w/2+1
        mean_f = torch.mean(x, dim=1, keepdim=True)  # b 1 h w/2+1
        fused_f = torch.cat([max_f, mean_f], dim=1)  # b 2 h w/2+1
        ffted_ = self.conv_layer_(fused_f)  # b 1 h w/2+1

        return ffted_

    def forward(self, x):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)

        # ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        # ffted = ffted.view((batch, -1,) + ffted.size()[3:])  # (batch, c*2, h, w/2+1)

        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted_ = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        if self.enhanced:
            ffted_ = self.enhanced_feat(ffted_)
        ffted_ = self.relu(ffted_)

        # ffted_ = self.conv_layer_(ffted_)  

        ffted = ffted * ffted_  # (batch, c*2, h, w/2+1)

        # ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = ffted.view((batch, 2, -1,) + ffted.size()[2:]).permute(0, 2, 3, 4, 1).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output


class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho', groups=2):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, 
                                          embed_dim * 2, 
                                          1, 1, 0,
                                          groups=groups)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)

        
         
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)   

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)

        # ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = ffted.view((batch, 2, -1,) + ffted.size()[2:]).permute(0, 2, 3, 4, 1).contiguous()  # (batch,c, t, h, w/2+1, 2)  
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output


class SpectralTransform(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 last_conv=False, 
                 func='Wave', 
                 dwt_device=None, 
                 if_print=False, layer_index=0):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.last_conv = last_conv
        self.func = func
        self.if_print = if_print
        self.layer_index = layer_index

        if 'Wave' in func:
            scale = 1
        else:
            scale = 2    

        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // scale, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = torch.nn.Conv2d(embed_dim // scale, embed_dim, 1, 1, 0)

        if func == 'Hardmard':
            self.fu = FourierUnit_H(embed_dim//scale,
                                    if_sparse=True,
                                    groups=2,
                                    # enhanced=True  
                                    enhanced=False  
                                    )        
        elif func == 'Vanilla':
            # self.fu = FourierUnit(embed_dim // 2, groups=1)   
            self.fu = FourierUnit(embed_dim // 2)
        elif func == 'Wave':
            self.fu = WaveLet(embed_dim//scale, device=dwt_device)
            # self.fu = WaveLet_(embed_dim//scale, device=dwt_device)
        elif func == 'Wave_en':
            self.fu = WaveLet_ELL(embed_dim//scale, 
                                  device=dwt_device,
                                  position=False,
                                  recons='idwt',  
                                  # recons='pixelshuffle', 
                                  # mode='hardmard'  
                                  mode='reweights',
                                  enhanced=False  
                                  )

             
        else:
            raise Exception('Please input correct choice.')    

        if self.last_conv:
            self.last_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x):

        x = self.conv1(x)

        if self.if_print:
            print_intermediate_feature(x.permute(0, 2, 3, 1).contiguous(), name=f'previous_layer_{self.layer_index}_{self.func}')
            if self.layer_index == 'shallow':
                print_intermediate_features(x.permute(0, 2, 3, 1).contiguous(), name=f'previous_layer_{self.layer_index}_{self.func}')

        output = self.fu(x)

        if self.if_print:
            print_intermediate_feature(output.permute(0, 2, 3, 1).contiguous(), name=f'future_layer_{self.layer_index}_{self.func}')
            if self.layer_index == 'shallow':
                print_intermediate_features(output.permute(0, 2, 3, 1).contiguous(), name=f'future_layer_{self.layer_index}_{self.func}')

        if 'Wave' in self.func:
            output = self.conv2(x * torch.sigmoid(output))
        else:    
            output = self.conv2(x + output)
        
        if self.last_conv:
            output = self.last_conv(output)
        return output


class WaveLet(nn.Module):
    def __init__(self, dim=64, n_groups=1, device=None):
        super().__init__()
        self.dim = dim

        if device is not None:
            device_ids = torch.cuda.current_device()
            device = torch.device(f'cuda:{device_ids}')
        else:
            device = torch.device('cpu')    
        
        self.wt = DWTForward(J=1, mode='zero', wave='haar').to(device)
        self.iwt = DWTInverse(mode='zero', wave='haar').to(device)

        self.conv_layer0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(dim*3, dim*3, 1, 1, 0, groups=n_groups),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.Conv2d(dim*3, dim*3, 1, 1, 0, groups=n_groups)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        yll, yh = self.wt(x)
        ylh = yh[0][:, :, 0, ...]
        yhl = yh[0][:, :, 1, ...]
        yhh = yh[0][:, :, 2, ...]

        yl = self.conv_layer0(yll)
        yh = torch.cat([ylh, yhl, yhh], dim=1)
        yh = self.conv_layer1(yh)

        ylh = yh[:, :self.dim, ...].contiguous()
        yhl = yh[:, self.dim:2*self.dim, ...].contiguous()
        yhh = yh[:, -self.dim:, ...].contiguous()
        yh = torch.stack([ylh, yhl, yhh], dim=2)

        y = self.iwt((yl, [yh]))

        return x + y


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim*3, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Enhanced_kv(nn.Module):
    def __init__(self, embed_dim, residual=False):
        super().__init__()
        self.res = residual
        self.conv = nn.Conv2d(embed_dim, 
                              embed_dim, 
                              3, 1, 1, 
                              groups=embed_dim)
              

    def forward(self, x):
        b, c, h, w = x.shape
        short_x = x

        # enhanced
        f = self.conv(x)
        x = x * f

        # dwconv
        # x = self.conv(x)

        if self.res:
            x = short_x + x
                    
        return x

# Enhanced LL sub-band
class WaveLet_ELL(nn.Module):
    def __init__(self, 
                 dim=64, 
                 n_groups=1, 
                 device=None, 
                 head=9, 
                 position=False,
                 recons='idwt',
                 mode='hardmard',
                 enhanced=False):
        super().__init__()
        self.dim = dim
        self.h = head
        self.head_dim = dim // head
        self.scale = self.head_dim ** -0.5
        self.recons = recons
        self.mode = mode
        self.pos = position
        self.enhanced = enhanced

        if device is not None:
            device_ids = torch.cuda.current_device()
            device = torch.device(f'cuda:{device_ids}')
        else:
            device = torch.device('cpu')    
        
        self.wt = DWTForward(J=1, mode='zero', wave='haar').to(device)
        if recons == 'idwt':
            self.iwt = DWTInverse(mode='zero', wave='haar').to(device)
        elif recons == 'pixelshuffle':    
            self.iwt = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(dim, dim, 1, 1, 0)
            )
        else:
            raise Exception('please input correct choice.')    

        self.conv_layer_q = nn.Sequential(
            # nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        self.conv_layer_k = nn.Sequential(
            # nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            Enhanced_kv(dim),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        self.conv_layer_v = nn.Sequential(
            # nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            Enhanced_kv(dim),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(dim*3, dim*3, 1, 1, 0, groups=n_groups),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.Conv2d(dim*3, dim*3, 1, 1, 0, groups=n_groups)  
        )

        if self.mode == 'reweights':
            self.reweight = Mlp(dim, dim // 4, dim * 3)  

        if position:
            self.position_coding = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)  
        
        if enhanced:
            self.high_freq_e = nn.Sequential(
                nn.Conv2d(2, 1, 1, 1, 0),
                nn.Sigmoid()
            )    

    def get_std(self, x, mean):
        b, c, h, w = x.shape

        std = (x - mean).pow(2).mean(dim=1, keepdim=True)
        std = std.sqrt()

        return std

    def enhanced_feat(self, x, ll=None, yh_=None):
        b, c, h, w = x.shape

        detail_coffi = x
        if ll is not None:
            x = torch.cat([ll, x], dim=1)

        max_f = torch.max(x, dim=1, keepdim=True)[0]  # b 1 h w
        mean_f = torch.mean(x, dim=1, keepdim=True)  # b 1 h w
        mean_f = self.get_std(x, mean_f)  # b 1 h w
        fused_f = torch.cat([max_f, mean_f], dim=1)  # b 2 h w
        ffted = self.high_freq_e(fused_f)  # b 1 h w/2+1

        if yh_ is None:
            return ffted * detail_coffi
        else:
            return ffted * yh_

    def attention(self, q, kv):
        B, C, H, W = q.shape
        N = H * W
        q = self.conv_layer_q(q).permute(0, 2, 3, 1).contiguous()
        k = self.conv_layer_k(kv).permute(0, 2, 3, 1).contiguous()
        v = self.conv_layer_v(kv)

        if self.pos:
            position = self.position_coding(v)  # b c h w
        
        v = v.permute(0, 2, 3, 1).contiguous()

        q = q.view(B, N, self.h, self.head_dim).permute(0, 2, 1, 3).contiguous() * self.scale
        k = k.view(B, N, self.h, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, self.h, self.head_dim).permute(0, 2, 1, 3).contiguous()  # b h n d

        attn = q @ k.transpose(-2, -1)  # b h n n
        attn = attn.softmax(dim=-1)
        out = attn @ v

        if self.pos:
            out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W) + position  
        else:
            out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)  

        out = self.proj(out)
        return out

    def forward_features_reweights(self, yll, ylh, yhl, yhh, shape):
        B, C, H, W = shape

        a = (ylh + yhl + yhh).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).contiguous().softmax(dim=0).unsqueeze(3).unsqueeze(3)
        fused_feat = ylh * a[0] + yhl * a[1] + yhh * a[2]
        yl = self.attention(yll, fused_feat)  

        yh_ = torch.cat([ylh, yhl, yhh], dim=1)
        yh = self.conv_layer1(yh_)

        if self.enhanced:
            yh = self.enhanced_feat(yh, yll)  
            # yh = self.enhanced_feat(yh, yll, yh_)

        ylh = yh[:, :self.dim, ...].contiguous()
        yhl = yh[:, self.dim:2*self.dim, ...].contiguous()
        yhh = yh[:, -self.dim:, ...].contiguous()

        return yl, ylh, yhl, yhh

    def forward_features_hardmard(self, yll, ylh, yhl, yhh, shape):
        B, C, H, W = shape

        yh = torch.cat([ylh, yhl, yhh], dim=1)
        yh = self.conv_layer1(yh)
        ylh = yh[:, :self.dim, ...].contiguous()
        yhl = yh[:, self.dim:2*self.dim, ...].contiguous()
        yhh = yh[:, -self.dim:, ...].contiguous()

        fused_feat = ylh * yhl * yhh
        yl = self.attention(yll, fused_feat)

        return yl, ylh, yhl, yhh
    
    def forward(self, x):
        B, C, H, W = x.shape

        yll, yh = self.wt(x)
        ylh = yh[0][:, :, 0, ...]
        yhl = yh[0][:, :, 1, ...]
        yhh = yh[0][:, :, 2, ...]

        if self.mode == 'hardmard':
            yl, ylh, yhl, yhh = self.forward_features_hardmard(yll, ylh, yhl, yhh, x.shape)
        elif self.mode == 'reweights':
            yl, ylh, yhl, yhh = self.forward_features_reweights(yll, ylh, yhl, yhh, x.shape)
        else:
            raise Exception('Please input right choice.')        
        
        # 反变换
        if self.recons == 'idwt':
            yh = torch.stack([ylh, yhl, yhh], dim=2)
            y = self.iwt((yl, [yh]))
        elif self.recons == 'pixelshuffle':
            # pixelshuffle
            y = torch.cat([yl, ylh, yhl, yhh], dim=1) 
            y = self.iwt(y)  
        else:
            raise Exception('please input correct choice.')    

        return x + y


class ResB(nn.Module):
    def __init__(self, embed_dim, reduce=1):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduce, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // reduce, embed_dim, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        return out + x


class DWB(nn.Module):
    def __init__(self, embed_dim, reduce=1):
        super(DWB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduce, 3, 1, 1, groups=embed_dim),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(embed_dim // reduce, embed_dim, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        return out + x


class FTB(nn.Module):
    def __init__(self,
                 layer_index=0, 
                 embed_dim=180,
                 mode='wf',  # wf: wave and fourier      w:wave only      f:fourier only
                 func0='Wave_en',
                 func1='Hardmard', 
                 reduce=1, 
                 layerscale_value=1e-4,
                 mlp_ratio=2, 
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 dwt_device=None,
                 if_print=False
                 ):
        super(FTB, self).__init__()
        self.mode = mode
        assert mode == 'wf' or mode == 'w' or mode == 'f', 'only three choices.'

        if mode == 'wf' or mode == 'w':
            self.norm0 = norm_layer(embed_dim)
            self.S0 = DWB(embed_dim, reduce)
            self.W = SpectralTransform(embed_dim,
                                       func=func0, 
                                       dwt_device=dwt_device, 
                                       if_print=if_print, 
                                       layer_index=layer_index)  
            self.fusion0 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

        if mode == 'wf' or mode == 'f':
            self.norm1 = norm_layer(embed_dim)
            self.S1 = DWB(embed_dim, reduce)
            # self.F = SpectralTransform(embed_dim, func='Vanilla', if_print=if_print, layer_index=layer_index) 
            # self.F = SpectralTransform(embed_dim, func='Dynamic')
            self.F = SpectralTransform(embed_dim, 
                                       func=func1, 
                                       if_print=if_print, 
                                       layer_index=layer_index)  
            self.fusion1 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

        # self.esa = DiVA_attention(embed_dim, esa_type='var')  
        self.esa = nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.if_gamma_f, self.if_gamma_w = False, False
        if mode == 'wf':
            self.if_gamma_f = True
            self.if_gamma_w = False
            # self.gamma_w = nn.Parameter(layerscale_value * torch.ones((embed_dim)), requires_grad=True)
            self.gamma_f = nn.Parameter(layerscale_value * torch.ones((embed_dim)), requires_grad=True)
        elif mode == 'f':
            self.if_gamma_f = False
            # self.gamma_f = nn.Parameter(layerscale_value * torch.ones((embed_dim)), requires_grad=True)
        elif mode == 'w':
            self.if_gamma_w = False
            # self.gamma_w = nn.Parameter(layerscale_value * torch.ones((embed_dim)), requires_grad=True)   
        else:     
            raise Exception('Please input correct choice.')
           
    def forward_wavelet(self, x):
        B, H, W, C = x.shape

        # WT // CNN
        short_x = x
        x = self.norm0(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        s = self.S0(x)
        w = self.W(x)
        out = torch.cat([s, w], dim=1)
        out = self.fusion0(out).permute(0, 2, 3, 1).contiguous()

        if self.if_gamma_w:
            x = (short_x * self.gamma_w) + self.drop_path(out)
        else:    
            x = short_x + self.drop_path(out)

        return x

    def forward_fourier(self, x):
        B, H, W, C = x.shape

        # FFT // CNN
        short_x = x
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        s = self.S1(x)
        f = self.F(x)
        out = torch.cat([s, f], dim=1)
        out = self.fusion1(out).permute(0, 2, 3, 1).contiguous()

        if self.if_gamma_f:
            x = (short_x * self.gamma_f) + self.drop_path(out)
        else:    
            x = short_x + self.drop_path(out)

        return x

    def forward(self, x):
        B, H, W, C = x.shape

        original_x = x
        
        if self.mode == 'wf' or self.mode == 'w':
            x = self.forward_wavelet(x)

        if self.mode == 'wf' or self.mode == 'f':    
            x = self.forward_fourier(x)
        
        # x = self.esa(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = original_x + x
        x = self.esa(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x
    