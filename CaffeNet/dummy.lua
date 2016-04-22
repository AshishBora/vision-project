dofile('torchModel_BC.lua')

-- read and preprocess dummy images
imFolder = '../testImages/'
path1 = imFolder .. 'cat.jpg'
path2 = imFolder .. 'Squirrel_posing.jpg'
label = torch.bernoulli() + 1
target = 2*label-3

image_data = {}
image_data[1] = preprocess(path1)
image_data[2] = preprocess(path2)
image_data[3] = image_data[label]:clone()

-- generate a dummy question
ques = torch.Tensor(1000):fill(0)
ques[285] = 1

-- dummy forward propagation through the model
output = BC_model:forward({image_data[1]:cuda(), image_data[2]:cuda(), image_data[3]:cuda(), ques:cuda()})
print(output[1][1], output[2][1])

-- dummy training
crit = nn.MarginRankingCriterion(0.1)
lr = 0.01
print(target)
for i = 1, 5 do
	trainStep(BC_model, {image_data[1]:cuda(), image_data[2]:cuda(), image_data[3]:cuda(), ques:cuda()}, target, crit, lr)
end