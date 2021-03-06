import torch
import torch.nn as nn
import poptorch
import numpy as np
import pdb

def get_options():
    opts = poptorch.Options()
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(1)
    opts.replicationFactor(1)
    opts.Training.gradientAccumulation(1)
    opts.setExecutionStrategy(poptorch.ShardedExecution())
    opts.outputMode(poptorch.OutputMode.All)  # Return all results from IPU to host
    opts.randomSeed(42)
    np.random.seed(42)
    return opts

class Modelbn(nn.Module):
    def __init__(self, c) -> None:
        super().__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, 255, 1) for x in [16, 32, 64])  # output conv

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return



if __name__ == "__main__":
    torch.manual_seed(0)
    inputs = torch.rand(1,3,2,2)
    model = Modelbn(3)

    # print('************ train *************')
    # model = model.train() # comment or uncomment this line

    # print('************ eval *************')
    # model = model.eval() # comment or uncomment this line

    opts = get_options()
    model = poptorch.inferenceModel(model, opts)

    print(model.bn.running_mean)
    print(model.bn.running_var)

    print("inputs ", inputs.view(-1))
    print("inputs mean ", inputs.mean())

    outputs = model(inputs)

    print(model.state_dict()['bn.running_mean'])
    print(model.state_dict()['bn.running_var'])

    print("outputs ", outputs.view(-1))
    print("outputs mean ", outputs.mean())

    pdb.set_trace()
    print('done')
    