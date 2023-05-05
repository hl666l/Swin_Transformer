import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

"""
PatchEmbed --->PatchMerging--->LN --->W-MSA(窗口自注意力机制，将一张图片分成几个等大小的窗口，每个窗口内进行注意力机制) 

---> LN ---> MLP

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
        """
        :param dim: 输入通道的数量
        :param window_size: window的长和宽
        :param num_heads: heads的数量
        :param qkv_bias: 如果为True，则向query, key, value添加一个可学习的偏差。默认值: True
        :param qk_scale:如果设置，覆盖head_dim ** -0.5的默认qk值
        :param atten_drop: attention weight丢弃率，默认: 0.0
        :param proj_drop: output的丢弃率. 默认:: 0.0
        """
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size  # MH，MW
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义相对位置偏差的参数列表（2*Mh-1 * 2*Mw-1, num_heads）
        self.relative_positive_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        # 获取窗口内每个token的成对相对位置索引
        # 获取feature map 的长宽，然后生成 长宽的每一个坐标
        """
        获取窗口内每个token的成对相对位置索引
        获取feature map 的长宽，然后生成 长宽的每一个坐标
        coords_h,coords_w两个tensor中包含图像内的每个像素的x,y坐标
        """
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        """
        将之前获取的x,y坐标拼接成一个矩阵，这样就形成了一张图片所有像素的位置矩阵 [2, h, w]
        """
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  # 沿h这个维度进行展平，[2,h*w] 绝对位置索引
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        # [2, Mh*Mw, Mh*Mw] 得到相对位置索引的矩阵。 以每一个像素作为参考点 - 当前feature map/window当中所有的像素点绝对位置索引 = 得到相对位置索引的矩阵
        # coords_flatten[:, :, None] 按w维度 每一行的元素复制
        # coords_flatten[:, None, :] 按h维度 每一行元素整体复制
        """
         [:, :, None] None表示对数组维度的扩充，为什么要扩充？
         [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
         [2, Mh*Mw, Mh*Mw] 得到相对位置索引的矩阵。 
         以每一个像素作为参考点 - 当前feature map/window当中所有的像素点绝对位置索引 = 得到相对位置索引的矩阵
        """
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # permute: 将窗口按每个元素求得的相对位置索引组成矩阵
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # 二元索引-->一元索引
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [h*w, h*w]
        # 放到模型缓存中
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
        # 解析输入维度[batch_size * num_windows, mh*mw, total_embed_dim]
        B_, N, C = x.shape
        #
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        x = (attn @ v).transformer(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP():
    pass


class WindowProcess:
    pass


class WindowProcessReverse:
    pass


def window_reverse():
    pass


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., atten_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, fused_window_process=False):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim  # 输入维度
        self.input_resolution = input_resolution
        self.num_heads = num_heads  #
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, atten_drop=atten_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = (int(dim * mlp_ratio))
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)
                        )
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)
                        )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.fused_window_process)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

