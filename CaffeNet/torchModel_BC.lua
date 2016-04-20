require 'loadcaffe';
require 'image';
require 'cunn';
require 'cudnn';

dofile('preproc.lua')

-- load the model
model = torch.load('caffenet.t7')
model:evaluate()
model:cuda()


-- encoder1 gets its parameters from the pretraind model
-- take only upto layer 21 to get 4096 dimensional
-- feature vector for each image
encoder1 = nn.Sequential()
for i=1,21 do
    encoder1:add(model.modules[i]:clone())
end

-- parent module which will house the three encoders
-- We need three because we don't want to handle random
-- choice inside the network right now. We will enentually
-- incorporate this

encoder = nn.ParallelTable()
encoder:add(encoder1)
-- clone copies as well as shares the parameters when called with
-- extra arguments
encoder:add(encoder1:clone('weight','bias', 'gradWeight','gradBias')) --clone the encoder and share the weight, bias. Must also share the gradWeight and gradBias
encoder:add(encoder1:clone('weight','bias', 'gradWeight','gradBias')) --clone the encoder and share the weight, bias. Must also share the gradWeight and gradBias

-- put the encoder on cuda and in evalautaion mode
encoder:cuda()
encoder:evaluate()

-- GetCateg takes the encoded output given by the third encoder and
-- predicts the category
getCateg = nn.Sequential()
for i=22,24 do
    getCateg:add(model.modules[i]:clone())
end

B = nn.ParallelTable()
B:add(getCateg)
B:add()


B:cuda()
B:evaluate()

-- C takes the encoded output given by the first two encoders,
-- the question (one hot for now) and a confidence value, which is B's output


imFolder = '/work/04001/ashishb/maverick/vision-project/testImages/'
path = imFolder .. 'cat.jpg'
input = preprocess(path)
feat = encoder1:forward(input:cuda());
output = B:forward(feat:cuda());
print(output)
