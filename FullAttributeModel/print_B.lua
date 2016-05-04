require 'cudnn'
require 'torch'
require 'nngraph'

dofile('func_lib.lua')

B_model = torch.load('B_model.t7')
cudnn.convert(B_model, nn)
printTable(B_model.modules)