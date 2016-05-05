-- require 'loadcaffe';
require 'image';
require 'cunn';
require 'cudnn';
require 'nngraph';

dofile('extractors.lua')

function create_C(getFeat, getAttr)

	local image1 = nn.Identity()();
	local image2 = nn.Identity()();
	local question = nn.Identity()();
	local confidence = nn.Identity()();

	local getFeat1 = getFeat:clone()
	local getFeat2 = getFeat1:clone('weight','bias')
	local image_feat1 = getFeat1(image1)
	local image_feat2 = getFeat2(image2)

	local getAttr1 = getAttr:clone()
	local getAttr2 = getAttr1:clone('weight','bias')
	local attScores1 = getAttr1(image_feat1)
	local attScores2 = getAttr2(image_feat2)

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
	return nn.gModule({image1, image2, question, confidence}, {prob});
end

sun_ws = torch.load('sun_ws.t7')
getFeat = get_getFeat(sun_ws, true)
getAttr = get_getAttr(sun_ws, true)

C_model = create_C(getFeat, getAttr)
torch.save('C_model.t7', C_model)