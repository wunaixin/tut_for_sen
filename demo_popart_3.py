import popart
builder = popart.Builder()
# Build a simple graph
i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
o = builder.aiOnnx.add([i1, i2])
builder.addOutputTensor(o)
# Get the ONNX protobuf from the builder to pass to the Session
proto = builder.getModelProto()
# Create a runtime environment
anchors = {o : popart.AnchorReturnType("ALL")}
dataFlow = popart.DataFlow(1, anchors)
device = popart.DeviceManager().createCpuDevice()
# Create the session from the graph, data feed and device information
session = popart.InferenceSession(proto, dataFlow, device)
