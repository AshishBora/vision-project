require 'cudnn'
require 'torch'
require 'nngraph'
require 'nn'
require 'cunn'

-- require 'clnn'
-- require 'cltorch'

dofile('func_lib.lua')
dofile('preproc.lua')
dofile('extractors.lua')

num_im = 5618
num_im = 10 -- for debug

base_path = '/work/04001/ashishb/maverick/vision-project/data/SUN/SUN_WS/test/sun_ws_test_'
imPaths = {}
for i  = 1, num_im do
	imPaths[i] = base_path .. tostring(i) .. '.jpg'
end

B_model = torch.load('B_model.t7')
predictor = get_predictor(B_model)
predictor:cuda()

attr_pred = torch.Tensor(num_im, 42)
for i = 1, num_im do
	input = preprocess(imPaths[i], 0, 0)
	output = predictor:forward(input:cuda())
	-- print(predictor.modules[2].output)
	attr = predictor.modules[2].output:double()
	-- print(attr)
	attr_pred[i] = attr
end

print(attr_pred:size())