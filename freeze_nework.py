import torch.onnx
import torchvision
import torch
import numpy as np
from torch.autograd import Variable
from model_thermal_rgb import Discriminator, Generator

dummy_input_IR = Variable(torch.randn(1, 1, 160, 120))
dummy_input_RGB = Variable(torch.randn(1, 3, 640, 480))

if 1:
    model = Generator()
    model.load_state_dict(torch.load('results/ir_rgb_0519_TIGAN/g-best.pth'))
    model.eval()
    torch.onnx.export(model, (dummy_input_IR,dummy_input_RGB),'results/ir_rgb_0519_TIGAN/g-best.onnx',
                        input_names=["low_ir","rgb"],output_names=["high_ir"], opset_version=11)


if 0:
    model = Discriminator()
    model.load_state_dict(torch.load('results/ir_rgb_0519_TIGAN/d-best.pth'))
    model.eval()
    dummy_input_IR = Variable(torch.randn(1, 1, 96, 96))
    torch.onnx.export(model, (dummy_input_IR),'results/ir_rgb_0519_TIGAN/d-best.onnx',
                        input_names=["ir"],output_names=["probability"], opset_version=10)
