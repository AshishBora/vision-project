require 'cudnn'
require 'torch'
require 'nngraph'
require 'nn'
require 'cunn';

-- require 'clnn'
-- require 'cltorch'

dofile('func_lib.lua')
dofile('preproc.lua')

base_path = '/work/04001/ashishb/maverick/vision-project/data/SUN/SUN_WS/test/sun_ws_test_'
imPaths = {}
for i  = 1, 5618 do
	imPaths[i] = base_path .. tostring(i) .. '.jpg'
end

sun_ws = torch.load('sun_ws.t7')
input = preprocess(imPaths[1], 0, 0)

A = nn.Sequential()
A:add(sun_ws)
A:add(nn.SoftMax())

A:cuda()
print(A)

output = A:forward(input:cuda())
print(output)