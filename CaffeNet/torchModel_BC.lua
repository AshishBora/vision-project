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

	local input1 = nn.JoinTable(1)({image_feat1, question, confidence});
	local input2 = nn.JoinTable(1)({image_feat2, question, confidence});

	local slp1 = nn.Linear(input_size, 1);
	local slp2 = slp1:clone('weight','bias', 'gradWeight','gradBias');
	local y1 = slp1(input1);
	local y2 = slp2(input2);

	nngraph.annotateNodes();
	return nn.gModule({image_feat1, image_feat2, question, confidence}, {y1, y2});
end


function createFullModel(B_model, C_model, encoders)
	local image1 = nn.Identity()();
	local image2 = nn.Identity()();
	local image3 = nn.Identity()();
	local question = nn.Identity()();

	local image_feat1 = encoders[1](image1);
	local image_feat2 = encoders[2](image2);
	local image_feat3 = encoders[3](image3);

	local confidence = B_model({question, image_feat3});
	local scores = C_model({image_feat1, image_feat2, question, confidence});
	
	nngraph.annotateNodes();
	return nn.gModule({image1, image2, image3, question}, {scores});
end


function getEncoders(model)
	-- encoder1 gets its parameters from the pretrained model
	-- take only upto layer 21 to get 4096 dimensional
	-- feature vector for each image
	local encoder1 = nn.Sequential()
	for i = 1, 21 do
		encoder1:add(model.modules[i]:clone())
	end

	-- clone copies as well as shares the parameters when called with extra arguments

	--clone the encoder and share the weight, bias. Must also share the gradWeight and gradBias
	local encoder2 = encoder1:clone('weight','bias', 'gradWeight','gradBias') 
	local encoder3 = encoder1:clone('weight','bias', 'gradWeight','gradBias')
	
	return {encoder1, encoder2, encoder3}
end


function getGetCateg(model)
	-- GetCateg takes the encoded output given by the third encoder and
	-- predicts the category
	local getCateg = nn.Sequential()
	for i = 22, 24 do
		getCateg:add(model.modules[i]:clone())
	end
	return getCateg
end


-- load the model
model = torch.load('caffenet.t7')
model:evaluate()
model:cuda()


-- create models
encoders = getEncoders(model)
getCateg = getGetCateg(model)
B_model = createFullBModel(getCateg)
C_model = createCModel(5097)
BC_model = createFullModel(B_model, C_model, encoders)


-- put all models on cuda and in evalaute mode
for i = 1, 3 do
	encoders[i]:cuda()
	encoders[i]:evaluate()
end

getCateg:cuda()
getCateg:evaluate()

B_model:cuda()
B_model:evaluate()

C_model:cuda()
C_model:evaluate()

BC_model:cuda()
BC_model:evaluate()


-- read and preprocess images
imFolder = '../testImages/'
path1 = imFolder .. 'cat.jpg'
path2 = imFolder .. 'Squirrel_posing.jpg'
label = torch.bernoulli() + 1

image_data = {}
image_data[1] = preprocess(path1)
image_data[2] = preprocess(path2)
image_data[3] = image_data[label]:clone()


-- generate a fake question
ques = torch.Tensor(1000):fill(0)
ques[285] = 1


-- forward propagate through the model
output = BC_model:forward({image_data[1]:cuda(), image_data[2]:cuda(), image_data[3]:cuda(), ques:cuda()})
print(output[1])
print(output[2])