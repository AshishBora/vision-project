require 'cudnn'
require 'torch'
require 'nngraph'
require 'nn'
require 'cunn'
require 'image'

dofile('func_lib.lua')
dofile('preproc.lua')
dofile('extractors.lua')

num_im = 5618
base_path = '/work/04001/ashishb/maverick/vision-project/data/SUN/SUN_WS/test/sun_ws_test_'
imPaths = {}
for i  = 1, num_im do
	imPaths[i] = base_path .. tostring(i) .. '.jpg'
end

B_model = torch.load('B_model.t7')
predictor = get_predictor(B_model)
predictor:cuda()

attr_pred = torch.Tensor(num_im, 42)
batch_size = 400
max_num_batches = torch.floor(num_im / batch_size) + 1

for i = 1, max_num_batches do
	start = (i-1)*batch_size + 1
	stop = math.min(i*batch_size, num_im)
	print('proecessing', start, 'to', stop)
	batch = torch.Tensor(stop-start+1, 3, 227, 227)
	for j = start, stop do
		img = image.load(imPaths[j], 3, 'float')
		img = preprocess(img, 0, 0)
		img = img:double()
		batch[j-start+1] = img
	end
	output = predictor:forward(batch:cuda())
	attr = predictor.modules[2].output:double()
	attr_pred[{{start, stop},{1,42}}] = attr
end

print(attr_pred:size())
torch.save('attr_pred.t7', attr_pred)
