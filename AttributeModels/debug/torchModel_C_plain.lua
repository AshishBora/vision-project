require 'nngraph';

function createCModel(input_size)
	local image_feat1 = nn.Identity()();
	local image_feat2 = nn.Identity()();
	local question = nn.Identity()();
	local confidence = nn.Identity()();

	local input1 = nn.JoinTable(1)({image_feat1, question, confidence});
	local input2 = nn.JoinTable(1)({image_feat2, question, confidence});

	local mlp1 = nn.Sequential()
    mlp1:add(nn.Linear(input_size, 64))
    mlp1:add(nn.ReLU())
    mlp1:add(nn.Linear(64, 1))
	local mlp2 = mlp1:clone('weight','bias', 'gradWeight','gradBias');
	local y1 = mlp1(input1);
	local y2 = mlp2(input2);

	nngraph.annotateNodes();
	return nn.gModule({image_feat1, image_feat2, question, confidence}, {y1, y2});
end

C_model = createCModel(4096+42+1)
torch.save('C_model.t7', C_model)