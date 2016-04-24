require 'loadcaffe';
require 'image';
require 'cunn';
require 'cudnn';
require 'nngraph';

function createFullAModel()
	local image_feat1 = nn.Identity()();
        local image_feat2 = nn.Identity()();
	
        local image_concat = nn.JoinTable(1)({image_feat1, image_feat2});
	
	local fc6 = nn.Linear(8192, 1000)(image_concat)

	nngraph.annotateNodes();
	return nn.gModule({image_feat1, image_feat2}, {fc6});
end


function createFullBModel(getAttScores)
	local image_feat = nn.Identity()()
	local question_input = nn.Identity()()
	local image_output = getAttScores(image_feat)
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


--function createFullModel(A_model, B_model, C_model, encoders)
--        local image1 = nn.Identity()();
--       local image2 = nn.Identity()();
--        local image3 = nn.Identity()();
--
--        local image_feat1 = encoders[1](image1);
--        local image_feat2 = encoders[2](image2);
--        local image_feat3 = encoders[3](image3);
--
--	local question = A_model({image_feat1, image_feat2});
--        local confidence = B_model({question, image_feat3});
--        local scores = C_model({image_feat1, image_feat2, question, confidence});
--
--        nngraph.annotateNodes();
--        return nn.gModule({image1, image2, image3}, {scores});
--end


-- function getEncoders(model)
-- 	-- encoder1 gets its parameters from the pretrained model
-- 	-- take only upto layer 21 to get 4096 dimensional
-- 	-- feature vector for each image
-- 	local encoder1 = nn.Sequential()
-- 	for i = 1, 21 do
-- 		encoder1:add(model.modules[i]:clone())
-- 	end

-- 	-- clone copies as well as shares the parameters when called with extra arguments

-- 	--clone the encoder and share the weight, bias. Must also share the gradWeight and gradBias
-- 	local encoder2 = encoder1:clone('weight','bias', 'gradWeight','gradBias') 
-- 	local encoder3 = encoder1:clone('weight','bias', 'gradWeight','gradBias')
	
-- 	return {encoder1, encoder2, encoder3}
-- end


function getGetAttScores(model)
	-- GetCateg takes the encoded output given by the third encoder and
	-- predicts the category
	local getAttScores = nn.Sequential()
	for i = 21, 23 do
		getAttScores:add(model.modules[i]:clone())
	end
	getAttScores:add(nn.Sigmoid())
	return getAttScores
end


-- load the model
model = torch.load('sun_ws.t7')
-- print(model)
-- model:evaluate()
-- model:cuda()
-- model:float()

-- convert B_model to nn backend
-- cudnn.convert(model, nn)

-- create models
-- encoders = getEncoders(model)
getAttScores = getGetAttScores(model)
B_model = createFullBModel(getAttScores)
C_model = createCModel(4096+42+1)
-- C_model:float()
-- A_model = createAModel();

torch.save('B_model.t7', B_model)
torch.save('C_model.t7', C_model)