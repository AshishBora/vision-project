require 'loadcaffe';
require 'image';
require 'cunn';
require 'cudnn';

dofile('preproc.lua')

-- load the model
model = torch.load('caffenet.t7')
model:evaluate()
model:cuda()

conv = nn.Sequential()
for i=1,21 do
    conv:add(model.modules[i]:clone())
end
conv:cuda()
conv:evaluate()

B = nn.Sequential()
for i=22,24 do
    B:add(model.modules[i]:clone())
end
B:cuda()
B:evaluate()

imFolder = '/work/04001/ashishb/maverick/vision-project/testImages/'
path = imFolder .. 'cat.jpg'
input = preprocess(path)
feat = conv:forward(input:cuda());
output = B:forward(feat:cuda());
print(output)