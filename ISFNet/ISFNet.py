from tkinter import Scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sam2.build_sam import build_sam2
from sam2.modeling.position_encoding import PositionEmbeddingSine
from inter_slice_refinement import build_Inter_Slice_Refinement


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up= nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv = DoubleConv(in_channels*4, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class PatchExpand(nn.Module):
    def __init__(self, dim, scale,  norm_layer=nn.LayerNorm):
        super(PatchExpand, self).__init__()
        
        self.scale = scale
        self.dim = dim  # 输入的通道数 C
        self.norm = norm_layer(dim)
        
    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: x [B, C/2, 2H, 2W]
        """

        # Pixel Shuffle 操作，将空间分辨率放大 2 倍，并减小通道数
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p2) (w p1)', p1=self.scale, p2=self.scale)
        
        # 归一化
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        
        return x
    
class Maskdecoder(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Downsampleconv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

    
class PrevMaskEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(PrevMaskEncoder, self).__init__()
        self.prevmask_encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=out_channels)
        
        self.conv_pe = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x0 = self.prevmask_encoder(x)
        x_pe = self.position_encoding(x0)
        x_cat = x0 + x_pe
        x = self.conv_pe(x_cat)
        
        return x


class ISFNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(ISFNet, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.downsample = Downsampleconv(2160, 64)
        self.prevmask_encoder = PrevMaskEncoder(1, 8, 64)
        self.patchexpand1 = PatchExpand(18,8)
        self.patchexpand2 = PatchExpand(36,4)
        self.patchexpand3 = PatchExpand(72,2)
        
        self.up1 = (Up(72, 144))
        self.up2 = (Up(36, 72))
        self.up3 = (Up(18, 36))
        self.conv1 = nn.Conv2d(144, 64, kernel_size=3, padding=1, bias=False)
        self.maskdecoder = Maskdecoder(128, 64)
        self.side1 = nn.Conv2d(36, 1, kernel_size=1)
        self.side2 = nn.Conv2d(72, 1, kernel_size=1)
        self.head = nn.Conv2d(144, 1, kernel_size=1)
        
        opt = {
            'hidden_dim': 18 ,
            'value_dim': 18,
            'S_window':  [7, 7],
            'shared_proj': True ,
        }
        self.ISR = build_Inter_Slice_Refinement(opt)
        self.ISR_2 = build_Inter_Slice_Refinement(opt)


    def forward(self, x, prev_image=None, prev_mask=None):
        if prev_image is None:
            prev_image = x
        if prev_mask is None:
            b,c,h,w = x.shape
            prev_mask = torch.zeros(b, 1, h, w).cuda()
        x1, x2, x3, x4 = self.encoder(x)
        px1, px2, px3, px4 = self.encoder(prev_image)
        x2, x3, x4 = self.patchexpand3(x2), self.patchexpand2(x3), self.patchexpand1(x4)
        px4 = self.patchexpand1(px4)

        # resx2_up = F.interpolate(resx2, size=(88, 88), mode='bilinear', align_corners=False)
        # resx3_up = F.interpolate(resx3, size=(88, 88), mode='bilinear', align_corners=False)
        # resx4_up = F.interpolate(resx4, size=(88, 88), mode='bilinear', align_corners=False)
        # resx = self.downsample(torch.cat([resx1, resx2_up, resx3_up, resx4_up], dim=1))
        pmask = self.prevmask_encoder(prev_mask)
        pmask = F.interpolate(pmask, size=(88, 88), mode='bilinear', align_corners=False)
        val_in = px4 + pmask
        
        val_out = self.ISR(x4 , val_in)
        val_out = self.ISR_2(x4 , val_out)
        
       
        x = self.up3(val_out, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=4, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=4, mode='bilinear')
        x = self.up1(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        # prev_mask = self.position_encoding(prev_mask)
        
        
        return out1 , out2 , out


if __name__ == "__main__":
    with torch.no_grad():
        model = ISFNet().cuda()
        x = torch.randn(8, 3, 352, 352).cuda()
        out1,out2,out = model(x)
        print(out1.shape,out2.shape,out.shape)