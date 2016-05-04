require 'torch';

attr_pred = torch.load('attr_pred.t7')

local matio = require 'matio'
matio.save('probs_test.mat', attr_pred)