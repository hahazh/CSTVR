
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )

class ResidualBlock3D_NoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlock3D_NoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat,(3,3,3), 1, 1, bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, (3,3,3), 1, 1, bias=True)
        self.relu = nn.PReLU()


    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride,padding=1, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DepthwiseTransSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super(DepthwiseTransSeparableConv3d, self).__init__()
        self.depthwise = nn.ConvTranspose3d(
        in_channels=in_channels,        
        out_channels=in_channels,       
        kernel_size=(3, 3, 3), 
        stride=stride,      
        padding=(1, 1, 1),     #  
        output_padding=(0, 1, 1) ,#
        groups=in_channels
    )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    
    
class ResidualBlock3D_depthwise_NoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, stride=1):
        super(ResidualBlock3D_depthwise_NoBN, self).__init__()
        self.res_scale = 1
        self.conv = DepthwiseSeparableConv3d(num_feat,num_feat,stride)
        self.relu = nn.PReLU()


    def forward(self, x):
        identity = x
        #必须加上激活函数！！！
        # out = self.relu(self.conv(x))
        out = self.conv(x)
        return identity + out * self.res_scale


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self,num_feats = [24,36,54,72]):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, num_feats[0], 3, 2, 1), 
            convrelu(num_feats[0], num_feats[0], 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(num_feats[0], num_feats[1], 3, 2, 1), 
            convrelu(num_feats[1], num_feats[1], 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(num_feats[1], num_feats[2], 3, 2, 1), 
            convrelu(num_feats[2], num_feats[2], 3, 1, 1)
        )
      
        
    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        return f1, f2, f3



class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
    ::
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)
class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, hw)
        """
        return self.main(feat)





class SpaceTimePixelShuffle(nn.Module):
    def __init__(self, r, s):
        super(SpaceTimePixelShuffle, self).__init__()
        self.r = r  # Time upscaling factor
        self.s = s  # Spatial upscaling factor

    def forward(self, x):
        b, c, t, h, w = x.size()
        c_out = int( c / (self.r * self.s * self.s))

        # First, reshape to split the spatial and temporal dimensions
        x = x.view(b, c_out, self.r, self.s, self.s, t, h, w)
        
        # Next, permute to interleave the spatial and temporal dimensions 这个应该是对的
        # (b, c_out, self.r, self.s, self.s, t, h, w) -> (b, c_out, t,self.r,h,self.s, w,self.s)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        # Finally, reshape to the desired output shape
        x = x.view(b, c_out, self.r * t, self.s * h, self.s * w)
        return x
class SpaceTimePixelUnShuffle(nn.Module):
    def __init__(self, r, s):
        super(SpaceTimePixelUnShuffle, self).__init__()
        self.r = r  # Time downscaling factor
        self.s = s  # Spatial downscaling factor

    def forward(self, x):
        b, c, t, h, w = x.size()
        c_out = c * (self.r * self.s * self.s)

        # First, reshape to prepare for permutation
        x = x.view(b, c, t // self.r, self.r, h // self.s, self.s, w // self.s, self.s)

        # Next, permute to rearrange the spatial and temporal dimensions
        # (b, c, t ,r, h, s, w, s) ->(b,c,r,s,s,t,h,w)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()

        # Finally, reshape to the desired output shape
        x = x.view(b, c_out, t // self.r, h // self.s, w // self.s)

        return x
# Example usage:


if __name__=='__main__':
    pass



            
