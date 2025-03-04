import torch.nn as nn
from module.inv_module import HaarDownsampling,InvBlockExp,ST_Pixelshuffle_Downsampling

class my_InvNN(nn.Module):
	def __init__(self, mode = 'test',channel_in=3, channel_out=3, subnet_constructor=None, block_num=2, down_num=2):
		super(my_InvNN, self).__init__()
		self.mode = mode
		operations = []

		current_channel = channel_in
		for i in range(down_num):
			# b = HaarDownsampling(current_channel)
			b = ST_Pixelshuffle_Downsampling()
			operations.append(b)
			current_channel *= 4
			for j in range(block_num):
				channel_out = 3
				spilt_mode = 0
				b = InvBlockExp(subnet_constructor, current_channel, channel_out,clamp=1,spilt_mode=spilt_mode)
				operations.append(b)
		# print(operations)
		self.operations = nn.ModuleList(operations)
		# self.qua = Quantization()
	def forward(self, x, rev=False, cal_jacobian=False):
		out = x
		jacobian = 0

		if not rev:
			
			for op in self.operations:
				
				out = op.forward(out, rev)
				
				if cal_jacobian:
					jacobian += op.jacobian(out, rev)
		else:
			
			for op in reversed(self.operations):
				out = op.forward(out, rev)
				if cal_jacobian:
					jacobian += op.jacobian(out, rev)
		if cal_jacobian:
			return out, jacobian
		else:
			return out


