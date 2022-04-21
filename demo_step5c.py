import torch
import poptorch
import pdb  


class Network(torch.nn.Module):
  def __init__(self):
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

inputs = torch.randn((4, 5))
targets = torch.randn((4, 5))


model = Network()
lr = 0.1
optimizer2 = torch.optim.SGD(params=model.parameters(), lr=lr)

opts = poptorch.Options()
# opts.deviceIterations(4)
opts.deviceIterations(1)
iputrainmodel = poptorch.trainingModel(model, options=opts)
optimizer = poptorch.optim.SGD(params=model.parameters(), lr=lr)

pdb.set_trace()
for i in range(2000):
# for i in range(10):
  pred, loss = iputrainmodel(inputs, targets)
  iputrainmodel.setOptimizer(optimizer)
  print(f'{i}: pred={pred}, loss={loss}')

pdb.set_trace()
model2 = Network()
for i in range(2000):
# for i in range(10):
  optimizer2.zero_grad()
  pred2, loss2 = model(inputs, targets)
  # pred2, loss2 = model2(inputs, targets)
  loss2.backward()
  optimizer2.step()
  # optimizer2.zero_grad()
  print(f'{i}: pred2={pred2}, loss2={loss2}')

pdb.set_trace()
print('done')
