require 'cudnn'
require 'torch'
require 'nngraph'
require 'cunn';

dofile('func_lib.lua')

C_model = torch.load('C_model.t7')
cudnn.convert(C_model, nn)
printTable(C_model.modules)