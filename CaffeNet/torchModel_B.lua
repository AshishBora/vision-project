require 'loadcaffe';
require 'image';
require 'cunn';
require 'cudnn';
require 'nngraph';
require 'nn';

require 'nngraph';
require 'nn';

QUESTION_SIZE=1000

function createFullBModel(imageModel)
    local image_input = nn.Identity()()
    local question_input = nn.Identity()()
    local image_output = imageModel(image_input)
    local output = nn.DotProduct()({image_output, question_input})
    nngraph.annotateNodes()
    return nn.gModule({question_input, image_input}, {output})
end


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

imFolder = '/work/04009/abhi23/maverick/vision-project/testImages/'
path = imFolder .. 'cat.jpg'
input = preprocess(path)
feat = conv:forward(input:cuda());
output = B:forward(feat:cuda());
question_input = torch.zeros(QUESTION_SIZE);
question_input[1] = 1.0;

dot_net = nn.DotProduct()
dot_net:cuda()
dot_net:evaluate()
dot_net_output = dot_net:forward({output:cuda(), question_input:cuda()})
print 'Dot Net Output'
print(dot_net_output)

full_B = createFullBModel(model);
full_B:cuda();
full_B:evaluate();
full_B_output = full_B:forward({question_input:cuda(), input:cuda()})
print 'Full B Output'
print(full_B_output[1])

print(output)
