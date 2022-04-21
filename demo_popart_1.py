# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import torch.onnx
import torchvision
input_ = torch.FloatTensor(torch.randn(4, 3, 224, 224))
model = torchvision.models.alexnet(pretrained=True)
output_name = "output"
torch.onnx.export(model, input_, "alexnet.onnx", output_names=[output_name])
# Create a runtime environment
anchors = {output_name: popart.AnchorReturnType("All")}
dataFlow = popart.DataFlow(100, anchors)
device = popart.DeviceManager().createCpuDevice()
session = popart.InferenceSession("alexnet.onnx", dataFlow, device)
