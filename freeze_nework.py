import torch.onnx
import torchvision
import torch
import numpy as np
from torch.autograd import Variable
from model_thermal_rgb import Discriminator, Generator
#from model import Discriminator, Generator

dummy_input_IR = Variable(torch.randn(1, 1, 64, 64))
dummy_input_RGB = Variable(torch.randn(1, 3, 256, 256))

if 1:
    weight_file = 'results/2023-06-11-CycleGANSR2/g-best.pth'
    model = Generator(export_onnx=True)
    model.load_state_dict(torch.load(weight_file))
    model.eval()
    # Change ext from pth to onnx
    torch.onnx.export(model, (dummy_input_IR,dummy_input_RGB),weight_file.replace('pth','onnx'),
                        input_names=["low_ir","rgb"],output_names=["high_ir"], opset_version=11)

    d_model = Discriminator()
    d_model.load_state_dict(torch.load(weight_file.replace('g-best','d-best')))
    d_model.eval()
    dummy_input_IR = Variable(torch.randn(1, 1, 96, 96))
    torch.onnx.export(d_model, (dummy_input_IR),weight_file.replace('g-best.pth','d-best.onnx'),
                        input_names=["ir"],output_names=["probability"], opset_version=11)


if 0:
    model = Generator()
    model.load_state_dict(torch.load('results/ir_sim_0531/g-best.pth'))
    model.eval()
    torch.onnx.export(model, (dummy_input_IR),'results/ir_sim_0531/g-best.onnx',
                        input_names=["low_ir"],output_names=["high_ir"], opset_version=11)

if 0:
    model = Discriminator()
    model.load_state_dict(torch.load('results/ir_rgb_0519_TIGAN/d-best.pth'))
    model.eval()
    dummy_input_IR = Variable(torch.randn(1, 1, 96, 96))
    torch.onnx.export(model, (dummy_input_IR),'results/ir_rgb_0519_TIGAN/d-best.onnx',
                        input_names=["ir"],output_names=["probability"], opset_version=10)
