require 'cudnn';
require 'nngraph';
require 'cunn';

dofile('extractors.lua')

function create_B(getFeat, getAttr)
	
	local image = nn.Identity()()
	local question = nn.Identity()()

	local image_feat = getFeat(image)
	local image_attr = getAttr(image_feat)

	local score = nn.DotProduct()({image_attr, question})
	nngraph.annotateNodes();
	return nn.gModule({question, image}, {score})
end

sun_ws = torch.load('sun_ws.t7')
getFeat = get_getFeat(sun_ws, true)
getAttr = get_getAttr(sun_ws, true)

B_model = create_B(getFeat, getAttr)
torch.save('B_model.t7', B_model)