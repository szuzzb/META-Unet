import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import resnet34
import os
import math
from einops import rearrange

class CBR(nn.Module):
    """
    This class defines the 2d convolutional layer with batch normalization and PReLU activation
    """
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, d=1):
        super().__init__()
        padding = (kSize-1) // 2
        self.conv2d = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=(padding, padding), bias=False, groups=groups, dilation=d)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: the input feature map
        :return: the output feature map
        """
        output = self.conv2d(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class Mlp(nn.Module):
    """
    This class defines the Feed Forward Network (Multilayer perceptron)
    """
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

class Self_Attention(nn.Module):
    def __init__(self, dim, ratio_h=2, ratio_w=2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """This class defines the self-attention utilized in the Efficient Transformer block used in the global branch of the META module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            ratio_h (int, optional): the reduction ratio of the height used in efficient transformer block. Defaults to 2.
            ratio_w (int, optional): the reduction ratio of the width used in efficient transformer block.. Defaults to 2.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            proj_drop (_type_, optional): Defaults to 0..
        """
        super().__init__()
        self.s = int(ratio_h * ratio_w)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.ke = nn.Conv2d(dim, dim, kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w), bias=qkv_bias)
        self.ve = nn.Conv2d(dim, dim, kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w), bias=qkv_bias)
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.ke(k.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2).reshape(B, N // self.s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.ve(v.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2).reshape(B, N // self.s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.norm_k(k)
        v = self.norm_v(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ETransformer_block(nn.Module):

    def __init__(self, dim, ratio_h=2, ratio_w=2, num_heads=8, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_features=None, mlp_ratio=4.,):
        """This class defines the Efficient Transformer block used in the global branch of the META module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            ratio_h (int, optional): the reduction ratio of the height used in efficient transformer block. Defaults to 2.
            ratio_w (int, optional): the reduction ratio of the width used in efficient transformer block.. Defaults to 2.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            proj_drop (_type_, optional): Defaults to 0..
            act_layer (_type_, optional): the action function used in FFN. Defaults to nn.GELU.
            norm_layer (_type_, optional): Defaults to nn.LayerNorm.
            out_features (_type_, optional): Defaults to None.
            mlp_ratio (_type_, optional): Defaults to 4..
        """
        super().__init__()
        self.out_features = out_features
        self.norm1 = norm_layer(dim)
        self.attn = Self_Attention(
            dim, ratio_h, ratio_w, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.out_features:
            return self.mlp(self.norm2(x))
        else:
            return x + self.mlp(self.norm2(x))

class Self_Attention_local(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """This class defines the self-attention utilized in the Efficient Transformer block used in the local branch of the META module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            proj_drop (_type_, optional): Defaults to 0..
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, R, N, C = x.shape
        qkv = self.qkv(x).reshape(B, R, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(-1, -2).reshape(B, R, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ETransformer_block_local(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, num_heads=8, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_features=None, mlp_ratio=4.,):
        """This class defines the Efficient Transformer block used in the local branch of the META module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            drop (_type_, optional): Defaults to 0..
            act_layer (_type_, optional): Defaults to nn.GELU.
            norm_layer (_type_, optional): Defaults to nn.LayerNorm.
            out_features (_type_, optional): Defaults to None.
            mlp_ratio (_type_, optional): Defaults to 4..
        """
        super().__init__()
        self.out_features = out_features
        self.norm1 = norm_layer(dim)
        self.attn = Self_Attention_local(
            dim, qkv_bias=qkv_bias, qk_scale=qk_scale, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.out_features:
            return self.mlp(self.norm2(x))
        else:
            return x + self.mlp(self.norm2(x))

class META(nn.Module):
    def __init__(self, dim, ph=4, pw=4, ratio_h=2, ratio_w=2, num_heads=8, drop=0., attn_drop=0.):
        """this class defines the Multiscale Efficient Transformer Attention module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            ph (int, optional): the patch size of height in the local branch. Defaults to 4.
            pw (int, optional): the patch size of width in the local branch. Defaults to 4.
            ratio_h (int, optional): the reduction ratio of the height used in efficient transformer block. Defaults to 2.
            ratio_w (int, optional): the reduction ratio of the width used in efficient transformer block.. Defaults to 2.
            num_heads (int, optional): Defaults to 8.
            drop (_type_, optional): Defaults to 0..
            attn_drop (_type_, optional): Defaults to 0..
        """
        super().__init__()
        self.ph = ph
        self.pw = pw
        self.loc_attn = ETransformer_block_local(dim=dim, num_heads=num_heads, drop=drop, attn_drop=attn_drop)
        self.glo_attn = ETransformer_block(dim=dim, ratio_h=ratio_h, ratio_w=ratio_w, num_heads=num_heads, drop=drop, attn_drop=attn_drop)

    def forward(self, x, feature=False):
        b, c, h, w = x.shape
        loc_x = rearrange(x, 'b d (nh ph) (nw pw) -> b (nh nw) (ph pw) d', ph=self.ph, pw=self.pw)
        glo_x = x.flatten(2).transpose(1, 2)
        loc_y = self.loc_attn(loc_x)
        loc_y = rearrange(loc_y, 'b (nh nw) (ph pw) d -> b d (nh ph) (nw pw)', nh=h // self.ph, nw=w // self.pw,
                          ph=self.ph, pw=self.pw)
        glo_y = self.glo_attn(glo_x)
        glo_y = glo_y.transpose(1, 2).reshape(b, c, h, w)
        y = loc_y + glo_y
        y = torch.sigmoid(y)
        if feature:
            return loc_y, glo_y, x * y
        else:
            return x * y

class Seg_head(nn.Module):
    def __init__(self, nIn, nOut):
        """this class defines the seg-head used in the META-Unet

        Args:
            nIn (_type_): the number of channel dimension of the input feature map
            nOut (_type_): the number of channel dimension of the output feature map (i.e. the number of the classes)
        """
        super().__init__()
        self.conv1 = CBR(nIn=nIn, nOut=nIn, kSize=3)
        self.conv2 = nn.Conv2d(nIn, nOut, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = x + self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.conv2(x)

        return x

class META_Unet(nn.Module):
    def __init__(self, nIn=3, classes=2, p1=4, p2=4, p3=4):
        """this class defines the META-Unet

        Args:
            nIn (int, optional): the number of channel dimension of the input image. Defaults to 3.
            classes (int, optional): the number of the classes. Defaults to 2.
            p1 (int, optional): the patch size used in the first META module. Defaults to 4.
            p2 (int, optional): the patch size used in the second META module. Defaults to 4.
            p3 (int, optional): the patch size used in the third META module. Defaults to 4.
        """
        super().__init__()
        self.backbone = resnet34(pretrained=True)
        channel = [32, 64, 128, 256, 512]
        num_heads = 4

        self.proj4 = CBR(nIn=channel[1], nOut=channel[0], kSize=1)
        self.proj8 = CBR(nIn=channel[2], nOut=channel[0], kSize=1)
        self.proj16 = CBR(nIn=channel[3], nOut=channel[0], kSize=1)
        self.proj32 = CBR(nIn=channel[4], nOut=channel[0], kSize=1)

        self.mstf32_16 = META(dim=channel[0], ph=p1, pw=p1, ratio_h=4, ratio_w=4, num_heads=num_heads, drop=0., attn_drop=0.)
        self.mstf16_8 = META(dim=channel[0], ph=p2, pw=p2, ratio_h=8, ratio_w=8, num_heads=num_heads, drop=0., attn_drop=0.)
        self.mstf8_4 = META(dim=channel[0], ph=p3, pw=p3, ratio_h=8, ratio_w=8, num_heads=num_heads, drop=0., attn_drop=0.)


        self.seg_head = Seg_head(channel[0], classes)

    def feature_forward(self, x):
        attention_list = []
        _, feat4, feat8, feat16, feat32 = self.backbone(x)

        feat4 = self.proj4(feat4)
        feat8 = self.proj8(feat8)
        feat16 = self.proj16(feat16)
        feat32 = self.proj32(feat32)

        feat32 = F.interpolate(feat32, scale_factor=2, mode="bilinear", align_corners=True)
        loc_16, glo_16, feat16 = self.mstf32_16(feat16 + feat32, feature=True)
        feat16 = F.interpolate(feat16, scale_factor=2, mode="bilinear", align_corners=True)
        loc_8, glo_8, feat8 = self.mstf16_8(feat8 + feat16, feature=True)
        feat8 = F.interpolate(feat8, scale_factor=2, mode="bilinear", align_corners=True)
        loc_4, glo_4, feat4 = self.mstf8_4(feat4 + feat8, feature=True)

        # b, c, h, w = feat4.shape
        # feat4 = feat4.flatten(2).transpose(-1, -2)
        # feat4 = self.emts1(feat4)
        # feat4 = feat4.transpose(-1, -2).reshape(b, c, h, w)
        feat = self.seg_head(feat4)
        attention_list.append(loc_16)
        attention_list.append(glo_16)
        attention_list.append(loc_8)
        attention_list.append(glo_8)
        attention_list.append(loc_4)
        attention_list.append(glo_4)
        # feat4 = F.interpolate(feat4, scale_factor=4, mode="bilinear", align_corners=True)
        # feat = self.seg_head(feat4)

        return attention_list, feat

    def forward(self, x):
        _, feat4, feat8, feat16, feat32 = self.backbone(x)

        feat4 = self.proj4(feat4)
        feat8 = self.proj8(feat8)
        feat16 = self.proj16(feat16)
        feat32 = self.proj32(feat32)

        feat32 = F.interpolate(feat32, scale_factor=2, mode="bilinear", align_corners=True)
        feat16 = self.mstf32_16(feat16 + feat32)
        feat16 = F.interpolate(feat16, scale_factor=2, mode="bilinear", align_corners=True)
        feat8 = self.mstf16_8(feat8 + feat16)
        feat8 = F.interpolate(feat8, scale_factor=2, mode="bilinear", align_corners=True)
        feat4 = self.mstf8_4(feat4 + feat8)

        # b, c, h, w = feat4.shape
        # feat4 = feat4.flatten(2).transpose(-1, -2)
        # feat4 = self.emts1(feat4)
        # feat4 = feat4.transpose(-1, -2).reshape(b, c, h, w)
        feat = self.seg_head(feat4)

        # feat4 = F.interpolate(feat4, scale_factor=4, mode="bilinear", align_corners=True)
        # feat = self.seg_head(feat4)

        return feat


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    model = META_Unet()
    model.eval()
    inputs = torch.rand(size=(1, 3, 320, 320))
    outputs = model(inputs)
    print(outputs.shape)
    
    
