import torch
import torch.nn as nn
import torch.nn.functional as F

'''
YOLOv8 + CPNConvNeXtv2, CSCConvNeXtv2, ReNLANConvNeXtv2, C3_ConvNeXtv2, C2f_ConvNeXtv2" 改进
- 只需要加上对应改进的核心模块，该项目代码就可以直接运行各种`YOLOv8-xxx.yaml`网络配置文件，乐高式创新改进，一键运行即可
- 相关改进有报错等 可以支持答疑服务。详情见 ⭐⭐⭐   ⭐⭐⭐ 说明
'''

# ...code

class C2f_ConvNeXtv2(nn.Module):
    def __init__(self, c1, c3, c4, num_blocks=2):
        """
        C2f_ConvNeXtv2 Module

        Args:
            c1 (int): Number of input channels.
            c3 (int): Number of output channels.
            c4 (int): Expansion ratio for ConvNeXt block.
            num_blocks (int): Number of ConvNeXt blocks to stack.
        """
        super().__init__()

        # Split input channels into two branches
        self.split_conv = nn.Conv2d(c1, c3, kernel_size=1, stride=1, bias=False)
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(c3, c4) for _ in range(num_blocks)
        ])

        # Combine fused features
        self.fuse_conv = nn.Conv2d(c3 * 2, c3, kernel_size=1, stride=1, bias=False)
        self.norm = nn.BatchNorm2d(c3)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_split = self.split_conv(x)
        residual = x_split.clone()  # Keep the initial input for fusion
        for block in self.blocks:
            x_split = block(x_split)
        x_fused = torch.cat((x_split, residual), dim=1)  # Concatenate along channel axis
        x_out = self.fuse_conv(x_fused)
        return self.act(self.norm(x_out))

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, expansion):
        """
        ConvNeXt Block

        Args:
            channels (int): Number of input/output channels.
            expansion (int): Expansion factor for the MLP layer.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)  # Depthwise Conv
        self.norm = nn.LayerNorm([channels, 1, 1])  # Channel-wise normalization
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Linear(channels * expansion, channels)
        )

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C] for LayerNorm
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = self.mlp(x.flatten(2).transpose(1, 2)).transpose(1, 2).view_as(x)
        return x + shortcut


class CSCConvNeXtv2(nn.Module):
    def __init__(self, c1, c3, c4):
        super().__init__()
        # 🎈YOLOv8 + CPNConvNeXtv2, CSCConvNeXtv2, ReNLANConvNeXtv2, C3_ConvNeXtv2, C2f_ConvNeXtv2" 改进==👇'
        # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见
        pass

class C2f_ConvNeXtv2(nn.Module):
    def __init__(self, c1, c3, c4):
        super().__init__()


        # 🎈YOLOv8 + CPNConvNeXtv2, CSCConvNeXtv2, ReNLANConvNeXtv2, C3_ConvNeXtv2, C2f_ConvNeXtv2" 改进==👇'
        # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见
        pass

class C3_ConvNeXtv2(nn.Module):

    def __init__(self, c1, c3, c4):
        super().__init__()
        # 🎈YOLOv8 + CPNConvNeXtv2, CSCConvNeXtv2, ReNLANConvNeXtv2, C3_ConvNeXtv2, C2f_ConvNeXtv2" 改进==👇'
        # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见
        pass

class CPNConvNeXtv2(nn.Module):
    def __init__(self, c2, c3, c4):
        super().__init__()
        # 🎈YOLOv8 + CPNConvNeXtv2, CSCConvNeXtv2, ReNLANConvNeXtv2, C3_ConvNeXtv2, C2f_ConvNeXtv2" 改进==👇'
        # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见
        pass

class ReNLANConvNeXtv2(nn.Module):
    def __init__(self, c1, c3, c4):
        # 🎈YOLOv8 + CPNConvNeXtv2, CSCConvNeXtv2, ReNLANConvNeXtv2, C3_ConvNeXtv2, C2f_ConvNeXtv2" 改进==👇'
        # 👉获取所有Backbone主干、Neck融合等改进核心模块, 详情见
        pass
