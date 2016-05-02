require 'nngraph';

function create_A(getAttScores)
    local image_feat1 = nn.Identity()();
    local image_feat2 = nn.Identity()();

    local getAttScores1 = getAttScores:clone()
	local getAttScores2 = getAttScores1:clone('weight','bias', 'gradWeight','gradBias')

	local attScores1 = getAttScores1(image_feat1)
	local attScores2 = getAttScores2(image_feat2)

    local attScores11 = nn.View(-1, 42)(attScores1);
    local attScores22 = nn.View(-1, 42)(attScores2);

    local att_concat = nn.JoinTable(2)({attScores11, attScores22});
    local question = nn.Linear(42*2, 42)(att_concat)
    local question_norm = nn.SoftMax()(question)

    nngraph.annotateNodes();
    return nn.gModule({image_feat1, image_feat2}, {question_norm});
end

B_model = torch.load('B_model_nn.t7')
getAttScores = B_model.modules[2]

A_model = create_A(getAttScores)
torch.save('A_model.t7', A_model)