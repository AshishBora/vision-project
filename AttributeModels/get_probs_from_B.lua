require 'nngraph';

B_model = torch.load('B_model_nn.t7')
B_model:evaluate()
B_predictor = B_model.modules[2]
B_model:double()

feat_vecs = torch.load('feat_vecs_test.t7')
feat_vecs:double()

probs_test = B_predictor:forward(feat_vecs)

local matio = require 'matio'
matio.save('probs_test.mat', probs_test)