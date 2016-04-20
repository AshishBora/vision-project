require 'loadcaffe';
require 'image';
require 'cunn';
require 'cudnn';
require 'nngraph';

dofile('preproc.lua')

function createFullBModel(getCateg)
	local image_feat = nn.Identity()()
	local question_input = nn.Identity()()
	local image_output = getCateg(image_feat)
	local output = nn.DotProduct()({image_output, question_input})
	nngraph.annotateNodes();
	return nn.gModule({question_input, image_feat}, {output})
end


function createCModel(input_size)
	local image_feat1 = nn.Identity()();
	local image_feat2 = nn.Identity()();
	local question = nn.Identity()();
	local confidence = nn.Identity()();
	local input1 = nn.ConcatTable()({image_feat1, question, confidence});
	local input2 = nn.ConcatTable()({image_feat2, question, confidence});
	local slp1 = nn.Linear(input_size, 1);
	local slp2 = slp:clone('weight','bias', 'gradWeight','gradBias');
	local y1 = slp1(input1);
	local y2 = slp2(input2);
	nngraph.annotateNodes();
	return nngraph.gModule({image_feat1, image_feat2, question, confidence}, {y1, y2});
end


function createFullModel(B_model, C_model, encoder)
	local image1 = nn.Identity()();
	local image2 = nn.Identity()();
	local image3 = nn.Identity()();
	local question = nn.Identity()();

	local image_feat1, image_feat2, image_feat3 = encoder({image1, image2, image3})
	
	local confidence = B_model(question, image_feat1);
	local scores = C_model(image_feat2, image_feat3, question, confidence);
	
	nngraph.annotateNodes();
	return nngraph.gModel({image1, image2, image3, question}, {scores});
end


function getEncoder(model)
	-- encoder1 gets its parameters from the pretrained model
	-- take only upto layer 21 to get 4096 dimensional
	-- feature vector for each image
	encoder1 = nn.Sequential()
	for i = 1, 21 do
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
	return encoder
end


function getGetCateg(model)
	-- GetCateg takes the encoded output given by the third encoder and
	-- predicts the category
	getCateg = nn.Sequential()
	for i = 22, 24 do
		getCateg:add(model.modules[i]:clone())
	end
end

-- load the model
model = torch.load('caffenet.t7')
model:evaluate()
model:cuda()

-- create models
encoder = getEncoder(model)
getCateg = getGetCateg(model)
B_model = createFullBModel(getCateg)
C_model = createCModel(5097)
BC_model = createFullModel(B_model, C_model, encoder)

imFolder = '../testImages/'
path1 = imFolder .. 'cat.jpg'
path2 = imFolder .. 'Squirrel_posing.jpg'
label = torch.bernoulli() + 1

image_data = {}
image_data[1] = preprocess(path[1])
image_data[2] = preprocess(path[2])
image_data[3] = image_data[label]:clone()

ques = torch.Tensor(1000):fill(0)
ques[285] = 1

output = BC_model:forward(image_data[1],	image_data[2], image_data[3], question})
print(output)