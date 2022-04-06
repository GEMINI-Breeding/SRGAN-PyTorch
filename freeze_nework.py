import torch.onnx
import torchvision
import torch
import numpy as np
from torch.autograd import Variable
from model_thermal_rgb import Generator

dummy_input_IR = Variable(torch.randn(1, 1, 160, 120))
dummy_input_RGB = Variable(torch.randn(1, 3, 640, 480))


model = Generator()
model.load_state_dict(torch.load('results/ir_rgb_0403_gd_autocastoff/g-best.pth'))
model.eval()
if 1:
    torch.onnx.export(model, (dummy_input_IR,dummy_input_RGB),'results/ir_rgb_0403_gd_autocastoff/g-best.onnx',
                        input_names=["low_ir","rgb"],output_names=["high_ir"], opset_version=10)
else:
    input1 = torch.randn(1, 1, 160, 120)
    input2 = torch.randn(1, 1, 640, 480)
    dummy_input = [input1, input2]

    input_names = [ "LOW_IR", "RGB" ]
    output_names = [ "HIGH_IR" ]

    torch.onnx.export(model, dummy_input, 'results/ir_rgb_0403_gd_autocastoff/g-best.onnx', 
                    input_names=input_names, 
                    output_names=output_names,
                    export_params=True)
