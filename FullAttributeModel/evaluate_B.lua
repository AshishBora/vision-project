require 'cudnn'
require 'torch'
require 'nngraph'
require 'nn'

-- require 'clnn'
-- require 'cltorch'

dofile('func_lib.lua')
dofile('preproc.lua')

base_path = '/work/04001/ashishb/maverick/vision-project/data/SUN/SUN_WS/test/sun_ws_test_'
imPaths = {}
for i  = 1, 5618 do
	imPaths[i] = base_path .. tostring(i) .. '.jpg'
end

B_model = torch.load('B_model.t7')

input = preprocess(imPaths[1], 0, 0)
ques = torch.Tensor(42):fill(0)
ques[3] = 1

cudnn.convert(B_model, nn)
B_model:double()
-- ques:double()
-- input:double()

-- B_model:cuda()
-- output = B_model:forward({ques:cuda(), input:cuda()})

output = B_model:forward({ques:double(), input:double()})

-- print(B_model.modules[2].output)
-- print(B_model.modules[3].output)
print(B_model.modules[5].output)
print(output)