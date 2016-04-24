require 'loadcaffe';
require 'image';
require 'cunn';
require 'cudnn';
require 'nngraph';

function createFullModel(B_model, C_model)
	local image_feat1 = nn.Identity()();
	local image_feat2 = nn.Identity()();
	local image_feat3 = nn.Identity()();
	local question = nn.Identity()();

	local confidence = B_model({question, image_feat3});
	local scores = C_model({image_feat1, image_feat2, question, confidence});
	
	nngraph.annotateNodes();
	return nn.gModule({image_feat1, image_feat2, image_feat3, question}, {scores});
end

B_model = torch.load('B_model.t7')
C_model = torch.load('C_model.t7')

B_model:cuda()
B_model:evaluate()

C_model:cuda()
C_model:evaluate()

BC_model = createFullModel(B_model, C_model)
-- ABC_model = createFullModel(A_model, B_model, C_model, encoders);

-- put all models on cuda and in evalaute mode

BC_model:cuda()
BC_model:evaluate()

-- Use a typical generic gradient update function
function trainStep(model, input, target, criterion, learningRate)
	local pred = model:forward(input)
	local err = criterion:forward(pred, target)
	local gradCriterion = criterion:backward(pred, target)

	model:zeroGradParameters()
	model:backward(input, {gradCriterion[1]:cuda(), gradCriterion[2]:cuda()})
	model:updateParameters(learningRate)

	print('pred = ', pred[1][1], pred[2][1])
	print('err = ', err)
end

-- read preprocessed feature vectors
feat_vecs = torch.load('feat_vecs.t7')

-- generate random label
label = torch.bernoulli() + 1
target = 2*label-3

-- generate dummy data
image_feat = {}
image_feat[1] = feat_vecs[1]
image_feat[2] = feat_vecs[2]
image_feat[3] = image_feat[label]:clone()

-- generate a dummy question
ques = torch.Tensor(42):fill(0)
ques[3] = 1

-- dummy forward propagation through the model
output = BC_model:forward({image_feat[1]:cuda(), image_feat[2]:cuda(), image_feat[3]:cuda(), ques:cuda()})
print(output[1][1], output[2][1])

-- dummy training
crit = nn.MarginRankingCriterion(0.1)
lr = 0.01
print(target)
for i = 1, 5 do
	trainStep(BC_model, {image_feat[1]:cuda(), image_feat[2]:cuda(), image_feat[3]:cuda(), ques:cuda()}, target, crit, lr)
end