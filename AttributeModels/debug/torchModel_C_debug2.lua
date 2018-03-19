require 'nngraph';


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

function createCModel(B_model)
	local image_feat1 = nn.Identity()();
	local image_feat2 = nn.Identity()();
	local question = nn.Identity()();
	local confidence = nn.Identity()();

	local B_model_copy1 = B_model:clone()
	local B_model_copy2 = B_model_copy1:clone('weight','bias', 'gradWeight','gradBias')

	local conf_pred1 = B_model_copy1({question, image_feat1})	
	local conf_pred2 = B_model_copy2({question, image_feat2})
	
	local input = nn.JoinTable(1)({conf_pred1, conf_pred2, confidence});

	local y = nn.Linear(3, 1)(input);
	local prob = nn.Sigmoid()(y);

	nngraph.annotateNodes();
	return nn.gModule({image_feat1, image_feat2, question, confidence}, {prob});
end

B_model = torch.load('B_model_nn.t7')
getAttScores = B_model.modules[2]
C_model = createCModel(getAttScores)
torch.save('C_model_debug.t7', C_model)