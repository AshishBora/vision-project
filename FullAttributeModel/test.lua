require 'torch'
require 'nn'

A = nn.Sequential()
A:add(nn.Linear(100,10))
A:add(nn.SoftMax())

x = torch.rand(100)
out = A:forward(x)

print(out)