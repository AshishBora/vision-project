require 'nngraph';


function createCModel(getAttScores)
	local image_feat1 = nn.Identity()();
	local image_feat2 = nn.Identity()();
	local question = nn.Identity()();
	local confidence = nn.Identity()();

	local getAttScores1 = getAttScores:clone()
	local getAttScores2 = getAttScores1:clone('weight','bias', 'gradWeight','gradBias')

	local attScores1 = getAttScores1(image_feat1)
	local attScores2 = getAttScores2(image_feat2)

	local conf_pred1 = nn.DotProduct()({attScores1, question})
	local conf_pred2 = nn.DotProduct()({attScores2, question})
    
    local conf_pred11 = nn.View(-1, 1)(conf_pred1);
    local conf_pred22 = nn.View(-1, 1)(conf_pred2);
    local confidence11 = nn.View(-1, 1)(confidence);

	local input = nn.JoinTable(2)({conf_pred11, conf_pred22, confidence11});

	local y = nn.Sequential()
	y:add(nn.Linear(3, 2))
	y:add(nn.Abs())
	y:add(nn.Linear(2,1))
	y:add(nn.Sigmoid())
	
	local prob = y(input);

	nngraph.annotateNodes();
	return nn.gModule({image_feat1, image_feat2, question, confidence}, {prob});
end

B_model = torch.load('B_model_nn.t7')
getAttScores = B_model.modules[2]
C_model = createCModel(getAttScores)

C_model_old = torch.load('C_model__1500_init.t7')
C_model.modules[13].modules[1].weight = C_model_old.modules[10].modules[1].weight:clone()
C_model.modules[13].modules[3].weight = C_model_old.modules[10].modules[3].weight:clone()
C_model.modules[13].modules[1].bias = C_model_old.modules[10].modules[1].bias:clone()
C_model.modules[13].modules[3].bias = C_model_old.modules[10].modules[3].bias:clone()

-- divide weights to get in trainable area
-- div_fact = 0.3
-- C_model.modules[13].modules[1].weight:mul(div_fact)
-- C_model.modules[13].modules[3].weight:mul(div_fact)
-- C_model.modules[13].modules[1].bias:mul(div_fact)
-- C_model.modules[13].modules[3].bias:mul(div_fact^2)

torch.save('C_model_temp.t7', C_model)