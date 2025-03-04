import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.model_utils as mutil
class HaarDownsampling(nn.Module):
	def __init__(self, channel_in):
		super(HaarDownsampling, self).__init__()
		self.channel_in = channel_in

		self.haar_weights = torch.ones(4, 1, 2, 2)

		# H
		self.haar_weights[1, 0, 0, 1] = -1
		self.haar_weights[1, 0, 1, 1] = -1

		# V
		self.haar_weights[2, 0, 1, 0] = -1
		self.haar_weights[2, 0, 1, 1] = -1

		# D
		self.haar_weights[3, 0, 1, 0] = -1
		self.haar_weights[3, 0, 0, 1] = -1

		self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
		self.haar_weights = nn.Parameter(self.haar_weights)
		self.haar_weights.requires_grad = False

	def forward(self, x, rev=False):
		if not rev:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(1 / 16.)
			# x_tmp = x.new_zeros(x.shape[0], x.shape[1], x.shape[2]+1, x.shape[3]+1)
			# x_tmp = x[:,:,:-1,:-1]
			
			out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
			# print('out.shape')
			out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
			out = torch.transpose(out, 1, 2)
			out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
			return out
		else:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(16.)

			out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
			out = torch.transpose(out, 1, 2)
			out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
			return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

	def jacobian(self, x, rev=False):
		return self.last_jac
		
class Pixelshuffle_Downsampling(nn.Module):
	def __init__(self):
		super(Pixelshuffle_Downsampling, self).__init__()
		# self.ps_shuffle = nn.PixelShuffle(2)
		# self.ps_unshuffle = nn.PixelUnshuffle(2)
	def forward(self, x, rev=False):
		if not rev:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(1 / 16.)
			[B, C, H, W] = list(x.size())
			x = x.reshape(B, C, H//2, 2, W//2, 2)
			x = x.permute(0, 1, 3, 5, 2, 4)
			x = x.reshape(B, C*4, H//2, W//2)
			return x
			# out = self.ps_unshuffle(x)
		else:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(16.)
			[B, C, H, W] = list(x.size())
			x = x.reshape(B, C//4, 2, 2, H, W)
			x = x.permute(0, 1, 4, 2, 5, 3)
			x = x.reshape(B, C//4, H*2, W*2)
			return x
	def jacobian(self, x, rev=False):
		return self.last_jac

class ST_Pixelshuffle_Downsampling(nn.Module):
	def __init__(self):
		super(ST_Pixelshuffle_Downsampling, self).__init__()
		self.r = 1
		self.s = 2
	def forward(self, x, rev=False):
		if not rev:
			b, c, t, h, w = x.size()
			self.elements =  c*t*h*w
			self.last_jac = self.elements / 4 * np.log(1 / 16.)
			c_out = c * (self.r * self.s * self.s)
			x = x.view(b, c, t // self.r, self.r, h // self.s, self.s, w // self.s, self.s)
			# Next, permute to rearrange the spatial and temporal dimensions
			# (b, c, t ,r, h, s, w, s) ->(b,c,r,s,s,t,h,w)
			x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
			# Finally, reshape to the desired output shape
			x = x.view(b, c_out, t // self.r, h // self.s, w // self.s)
			return x
			# out = self.ps_unshuffle(x)
		else:
			b, c, t, h, w = x.size()
			self.elements =  c*t*h*w
			self.last_jac = self.elements / 4 * np.log(1 / 16.)
			c_out = int( c / (self.r * self.s * self.s))
			x = x.view(b, c_out, self.r, self.s, self.s, t, h, w)
			# (b, c_out, self.r, self.s, self.s, t, h, w) -> (b, c_out, t,self.r,h,self.s, w,self.s)
			x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
			x = x.view(b, c_out, self.r * t, self.s * h, self.s * w)
			return x


	def jacobian(self, x, rev=False):
		return self.last_jac


class InvBlockExp(nn.Module):
	def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.,spilt_mode = 0):
		super(InvBlockExp, self).__init__()

		self.split_len1 = channel_split_num
		self.split_len2 = channel_num - channel_split_num
		self.spilt_mode = spilt_mode
		self.clamp = clamp
		# pdb.set_trace()
		self.F = subnet_constructor(self.split_len2, self.split_len1)
		self.G = subnet_constructor(self.split_len1, self.split_len2)
		self.H = subnet_constructor(self.split_len1, self.split_len2)
	
	def forward(self, x,rev=False):
		if self.spilt_mode==1:
			x = torch.flip(x,dims=[1])

		x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

		
		if not rev:
			y1 = x1 + self.F(x2)
			self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
			y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
		else:
			self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
			y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
			y1 = x1 - self.F(y2)

		return torch.cat((y1, y2), 1)

	def jacobian(self, x, rev=False):
		if not rev:
			jac = torch.sum(self.s)
		else:
			jac = -torch.sum(self.s)
		

		return jac / x.shape[0]




class D2DTInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier',\
         gc=32, bias=True,INN_init = True,is_res = False):
        super(D2DTInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="3d"):
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
   
        return x5
class HaarTransform(nn.Module):
    def __init__(self, channel_in):
        super(HaarTransform, self).__init__()
        # Initialize the Haar weights
        haar_weights = torch.ones(4, 1, 2, 2)
        haar_weights[1, 0, 0, 1] = -1
        haar_weights[1, 0, 1, 1] = -1
        haar_weights[2, 0, 1, 0] = -1
        haar_weights[2, 0, 1, 1] = -1
        haar_weights[3, 0, 1, 0] = -1
        haar_weights[3, 0, 0, 1] = -1
        haar_weights = haar_weights.repeat(channel_in, 1, 1, 1)
        
        # Register the weights as a non-trainable parameter
        self.register_buffer('haar_weights', haar_weights)

    def forward(self, x):
        channel_in = x.shape[1]
        # Perform the Haar transformation using the registered weights
        out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=channel_in) / 4.0
        # Reshape the output tensor to move the sub-band coefficients next to the channel dimension
        out = out.view(x.shape[0], channel_in, 4, x.shape[2] // 2, x.shape[3] // 2)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(x.shape[0], channel_in * 4, x.shape[2] // 2, x.shape[3] // 2)
        return out


#转换
def Haartrans(x):
    channel_in = x.shape[1]
    haar_weights = torch.ones(4, 1, 2, 2).to(x.device)
    haar_weights[1, 0, 0, 1] = -1
    haar_weights[1, 0, 1, 1] = -1
    haar_weights[2, 0, 1, 0] = -1
    haar_weights[2, 0, 1, 1] = -1
    haar_weights[3, 0, 1, 0] = -1
    haar_weights[3, 0, 0, 1] = -1
    haar_weights = torch.cat([haar_weights] * channel_in, 0)
    haar_weights.requires_grad = False
    out = F.conv2d(x, haar_weights, bias=None, stride=2, groups=channel_in) / 4.0
    # print('out.shape')
    out = out.reshape([x.shape[0], channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
    out = torch.transpose(out, 1, 2)
    out = out.reshape([x.shape[0], channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
    return out

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

def Haartrans_back(x):
    channel_in = x.shape[1]//4
    haar_weights = torch.ones(4, 1, 2, 2).to(x.device)
    haar_weights[1, 0, 0, 1] = -1
    haar_weights[1, 0, 1, 1] = -1
    haar_weights[2, 0, 1, 0] = -1
    haar_weights[2, 0, 1, 1] = -1
    haar_weights[3, 0, 1, 0] = -1
    haar_weights[3, 0, 0, 1] = -1
    haar_weights = torch.cat([haar_weights] * channel_in, 0)
    haar_weights.requires_grad = False
    out = x.reshape([x.shape[0], 4, channel_in, x.shape[2], x.shape[3]])
    out = torch.transpose(out, 1, 2)
    out = out.reshape([x.shape[0], channel_in * 4, x.shape[2], x.shape[3]])
    return F.conv_transpose2d(out, haar_weights, bias=None, stride=2, groups=channel_in)

class D2DTInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier',\
         shortcut = True,gc=32, bias=True,INN_init = True,is_res = False):
        super(D2DTInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.shortcut = shortcut
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="3d"):
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if  self.shortcut:
            x5 = x+x5
        return x5


class D2DTInput_dense(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier',\
         gc=32, bias=True,INN_init = True,is_res = False):
        super(D2DTInput_dense, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (1,1,1), 1, (0,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="3d"):
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
   
        return x5