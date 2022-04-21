import torch
import poptorch
import pdb  #bm

#bm, cpu
# class Network2(torch.nn.Module):
#   def __init__(self) -> None:
#       super().__init__()
#       self.layer1 = torch.nn.Linear(5, 10)
#       self.layer2 = torch.nn.Linear(10, 5)
#       self.layer3 = torch.nn.Linear(5, 5)
#       self.layer4 = torch.nn.Linear(5, 5)
#       self.act = torch.nn.ReLU()
#       self.softmax = torch.nn.Softmax(dim=1)

#   def forward(self, x):
#     x = self.layer1(x)
#     x = self.act(x)
#     x = self.layer2(x)
#     x = self.act(x)
#     x = self.layer3(x)
#     x = self.act(x)
#     x = self.layer4(x)
#     x = self.act(x)
#     x = self.softmax(x)
#     return x

class Network2(torch.nn.Module):
  def __init__(self) -> None:
      super().__init__()
      self.layer1 = torch.nn.Linear(5, 10)
      self.layer2 = torch.nn.Linear(10, 5)
      self.layer3 = torch.nn.Linear(5, 5)
      self.layer4 = torch.nn.Linear(5, 5)
      self.act = torch.nn.ReLU()
      self.softmax = torch.nn.Softmax(dim=1)
      self.loss = torch.nn.MSELoss() 

  def forward(self, x, target=None):
    x = self.layer1(x)
    x = self.act(x)
    x = self.layer2(x)
    x = self.act(x)
    x = self.layer3(x)
    x = self.act(x)
    x = self.layer4(x)
    x = self.act(x)
    x = self.softmax(x)
    if self.training:
      return x, self.loss(x, target)
    else:
      return x


class Network(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = torch.nn.Linear(5, 10)
    self.layer2 = torch.nn.Linear(10, 5)
    self.layer3 = torch.nn.Linear(5, 5)
    self.layer4 = torch.nn.Linear(5, 5)
    self.act = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)
    self.loss = torch.nn.MSELoss()            #bm

  def forward(self, x, target=None):
    x = self.layer1(x)
    x = self.act(x)
    x = self.layer2(x)
    x = self.act(x)
    x = self.layer3(x)
    x = self.act(x)
    x = self.layer4(x)
    x = self.act(x)
    x = self.softmax(x)
    if self.training:
      return x, self.loss(x, target)
    else:
      return x

  # def forward(self, x):
    # # Explicit layers on a certain IPU
    # poptorch.Block.useAutoId()
    # with poptorch.Block(ipu_id=0):
    #   x = self.act(self.layer1(x))

    # with poptorch.Block(ipu_id=1):
    #   x = self.act(self.layer2(x))

    # with poptorch.Block(ipu_id=2):
    #   x = self.act(self.layer3(x))
    #   x = self.act(self.layer4(x))

    # with poptorch.Block(ipu_id=3):
    #   x = self.softmax(x)

    # return x


inputs = torch.randn((4, 5))
targets = torch.randn((4, 5))

model = Network()
opts = poptorch.Options()
# opts.deviceIterations(4)
opts.deviceIterations(1)

iputrainmodel = poptorch.trainingModel(model, options=opts)
pdb.set_trace()
for i in range(2000):
  pred, loss = iputrainmodel(inputs, targets)
  print(f'{i}: pred={pred}, loss={loss}')

# poptorch_model = poptorch.inferenceModel(model, options=opts)
# print(poptorch_model(torch.rand((4, 5))))
# out = poptorch_model(inputs)

# model2 = Network2()
# out2 = model2(inputs)

pdb.set_trace()
for i in range(2000):
  # pred2, loss2 = model2(inputs, targets)
  pred2, loss2 = model(inputs, targets)
  loss2.backward()
  print(f'{i}: pred2={pred2}, loss2={loss2}')

pdb.set_trace()
print('done')
