require 'loadcaffe';
require 'image';
require 'cunn';
require 'cudnn';
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

B_model = torch.load('B_model.t7')
getAttScores = B_model.modules[2]
C_model = createCModel(getAttScores)
torch.save('C_model.t7', C_model)
