require 'loadcaffe';
require 'image';
require 'cunn';
require 'cudnn';

dofile('preproc.lua')

-- load the model
model = torch.load('caffenet.t7')
model:evaluate()
model:cuda()

imFolder = '/work/04001/ashishb/maverick/vision-project/testImages/'
path = imFolder .. 'cat.jpg'
input = preprocess(path)
output = model:forward(input:cuda());
print(output)
