require 'nngraph';

function create_A()
    local image_feat1 = nn.Identity()();
    local image_feat2 = nn.Identity()();
    local image_concat = nn.JoinTable(1)({image_feat1, image_feat2});
    local question = nn.Linear(8192, 42)(image_concat)

    nngraph.annotateNodes();
    return nn.gModule({image_feat1, image_feat2}, {question});
end

A_model = create_A()
torch.save('A_model.t7', A_model)