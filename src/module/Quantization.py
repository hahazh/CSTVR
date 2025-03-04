import torch
import torch.nn as nn
import torch.nn.functional as F
class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)
    
class Quantize_ste(nn.Module):
    def __init__(self,min_val, max_val):
        super(Quantize_ste, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
    def forward(self, x):
        x_clipped_min = x + F.relu(self.min_val - x)
        x_clipped = x_clipped_min - F.relu(x_clipped_min - self.max_val)
        return  (torch.round(x_clipped*255.)/255. - x_clipped).detach() + x_clipped

class DifferentiableClipping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_ten, min_val, max_val):
        # Save for backward

        # in_ten = torch.clamp(in_ten, 0, 1)

        ctx.save_for_backward(in_ten)
        ctx.min_val = min_val
        ctx.max_val = max_val
        x_clipped_min = in_ten + F.relu(min_val - in_ten)
        x_clipped = x_clipped_min - F.relu(x_clipped_min - max_val)

        x_clipped = (x_clipped * 255.).round() / 255.
        return x_clipped

    @staticmethod
    def backward(ctx, grad_output):
        in_ten, = ctx.saved_tensors
        min_val = ctx.min_val
        max_val = ctx.max_val
        grad_input = grad_output.clone()
        grad_input[in_ten <= min_val] = 2
        grad_input[in_ten >= max_val] = 2
        # Return gradients for input and None for min_val and max_val since they are not tensors
        return grad_input, None, None

class DifferentiableQuantization(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super(DifferentiableQuantization, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input):
        return DifferentiableClipping.apply(input, self.min_val, self.max_val)


# Example usage
quantization_layer = DifferentiableQuantization(0.0, 255.0)
input_tensor = torch.randn(10, requires_grad=True)
output = quantization_layer(input_tensor)
output.backward(torch.ones_like(input_tensor))  # Backward to test the gradient computation
