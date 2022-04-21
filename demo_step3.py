import torch
import torchvision
import poptorch
import pdb   #bm

# Some dummy imagenet sized input.
picture_of_a_cat_here = torch.randn([1, 3, 224, 224])

# The model, in this case a MobileNet model with pretrained weights that comes
# canned with Pytorch.
model = torchvision.models.mobilenet_v2(pretrained=True)
model.train(False)

# Wrap in the PopTorch inference wrapper
inference_model = poptorch.inferenceModel(model)

# Execute on IPU.
pdb.set_trace()
out_tensor = inference_model(picture_of_a_cat_here)

# Get the top 5 ImageNet classes.
top_five_classes = torch.topk(torch.softmax(out_tensor, 1), 5)
print(top_five_classes)

# Try the same on native PyTorch
native_out = model(picture_of_a_cat_here)

native_top_five_classes = torch.topk(torch.softmax(native_out, 1), 5)

# Models should be very close to native output although some operations are
# numerically different and floating point differences can accumulate.
assert any(top_five_classes[1][0] == native_top_five_classes[1][0])
# inference_half_start
model = torch.nn.Linear(1, 10)

# Convert the parameters (weights) to halfs. Without doing so,
# the Linear parameters will automatically be cast to half, which allows
# training with float32 parameters but half tensors.
model.half()
t1 = torch.tensor([1.]).half()

opts = poptorch.Options()

inference_model = poptorch.inferenceModel(model, opts)
out = inference_model(t1)

assert out.dtype == torch.half
# inference_half_end

pdb.set_trace()
print('done')
