import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

"""
PatchEmbed --->PatchMerging--->

"""


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        """
        :param img_size:
        :param patch_size: 高宽要分成多少份
        :param in_chans: 输入通道
        :param embed_dim: 编码后输出通道
        :param norm_layer: 是否添加归一化层
        """
        super.__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 计算出 Patch token在长宽方向上的数量

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # ’//‘得到除数后取整
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        # 计算出patch的数量利用Patch token在长宽方向上的数量相乘的结果
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 划分后，每张patch的像素数量
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        """
        这个卷积中的核，步长能够恰好得到 56x56的图片
        故使用此种方法获得 96张 56x56的图片

        """
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 判断是否使用norm_layer，在这里我们没有应用
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 解析输入的维度 B是图片张数, C通道数, H高, W宽
        B, C, H, W = x.shape
        # 判断图像是否与设定图像一致，如果不一致会报错
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 经过一个卷积层来进行一个线性变换，并且在第二个维度上进行一个压平操作，维度为(B, C, Ph*Pw),后在进行一个维与二维的一个转置，维度为：(B Ph*Pw C)
        """
        注意 self.proj(x)获得的图片的shape是【B，96, 56,56】，经过flatten(2)，将图片的第三个通道拉直变成【B，96, 56X56】
        通过transpose(1, 2)交换第二，第三维度变成【B，56X56,96】
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)  # 判断是否有归一化操作
        return x  # x.shape=[B，56X56,96]


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)  # 重置x的shape
        """
        x[:, 0::2, 0::2, :]表示的含义是一张图片高从0到末尾，步长为2取数。
        故一张图片通过这种方法共取出四张图片，每张图片大小 [H/2,W/2]。
        """
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        """
        torch.cat([x0, x1, x2, x3], -1)将四张图片叠加起来，因此得到 [B H/2 W/2 4*C]
        x.view(B, -1, 4 * C) 给定两头的参数值，用-1占一个通道的位置，并告诉编译器自动确定-1位置的参数
        """
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    #  计算参数量
    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class WindowAttention(nn.Module):
    # 实现W-MSA，SW-MSA
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, atten_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_positive_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(atten_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_positive_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        :param x: input features with shape of (num_windows*B, N, C)
        :param mask:
        :return:
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


class SwinTransformerBlock():
    pass
