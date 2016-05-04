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

-- function to get an example for training C
function getCtrainExample(set, labels)

    -- randomly select two images from different classes
    local y = torch.randperm((#set)[1])

    local im_feat = {}
    im_feat[1] = set[y[1]]
    im_feat[2] = set[y[2]]

    -- randomly select one of those to be given to model B
    local label = torch.bernoulli() + 1
    im_feat[3] = im_feat[label]:clone()

    -- randomly select one of the images to ask about its class
    local ques = torch.Tensor(42):fill(0)
    ques[labels[y[torch.bernoulli() + 1]]] = 1

    -- target is probability of label = 2
    local target = label - 1

    -- Testing
    -- outfile:write(im1_Path)
    -- outfile:write(im2_Path)
    -- outfile:write(y[1], y[2])
    -- outfile:write(label)
    -- outfile:write(ques[y[1]], ques[y[2]])
    
    local input = {im_feat[1], im_feat[2], im_feat[3], ques}
    
    return {input, target}
end

function do_weight_decay(model, wd)
    lin_modules = model:findModules('nn.Linear');
    for i = 1,#lin_modules do
        m = torch.mul(lin_modules[i].weight, wd);
        lin_modules[i].gradWeight = lin_modules[i].gradWeight + m;
    end
end

function get_pred_err(prob, target)

    local pred = 2
    if(prob[1] < 0.5) then
        pred = 1
    end

    local pred_err = 0
    if pred ~= target+1 then
        pred_err = 1
    end

    return pred_err
end

-- Use a typical generic gradient update function
function accumulate(model, input, target, criterion, eval_criterion, batch_size, wd)
    local prob = model:forward(input)
    local loss = criterion:forward(prob, torch.Tensor{target})
    local gradCriterion = criterion:backward(prob, torch.Tensor{target})
    model:backward(input, gradCriterion, 1/batch_size)
    do_weight_decay(model, wd)
    local pred_err = get_pred_err(prob, target)
    -- print('prob = ', prob)
    return loss, pred_err
end


-- function to evalaute the model
function evalPerf(model, criterion, set, labels, iter)

    outfile = io.open("train_C.out", "a")
    outfile:write('Testing... ')
    outfile:close()

    local test_loss = 0
    local test_pred_err = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(3489208)

    for j = 1, iter do
        example = getCtrainExample(set, labels)
        input = example[1]
        target = example[2]
        local prob = model:forward({input[1], input[2], input[3], input[4]})
        local samp_loss = criterion:forward(prob, torch.Tensor{target})
        local pred_err = get_pred_err(prob, target)
        test_pred_err = test_pred_err + pred_err
        test_loss = test_loss + samp_loss
    end

    outfile = io.open("train_C.out", "a")
    outfile:write('average test_loss = ', test_loss/iter, ', ')
    outfile:write('average test_pred_err = ', test_pred_err/iter, '\n')
    outfile:close()
end


-- get some essential functions
-- outfile = io.open("train_C.out", "w")
-- outfile:write('Running string split... ')
-- dofile('string_split.lua')
-- outfile:write('done\n')
-- outfile:close()

-- outfile = io.open("train_C.out", "a")
-- outfile:write('Running getImPaths... ')
-- dofile('getImPaths.lua')
-- outfile:write('done\n')
-- outfile:close()

-- Laod the original model and creat BC model
outfile = io.open("train_C.out", "w")
outfile:write('Loading pretrained model... ')

B_model = torch.load('B_model_nn.t7')
C_model = torch.load('C_model__1500_init.t7')
BC_model = createFullModel(B_model, C_model)
-- ABC_model = createFullModel(A_model, B_model, C_model, encoders);

-- convert to double
BC_model:double()

outfile:write('done\n')
outfile:close()


-- read preprocessed feature vectors and labels
feat_vecs = torch.load('feat_vecs.t7')
labels = torch.load('labels.t7')

-- randomly shuffle data
feat_vecs_temp = feat_vecs:clone()
labels_temp = labels:clone()

y = torch.randperm((#feat_vecs)[1])
for i = 1, ((#feat_vecs)[1]) do
    feat_vecs[i] = feat_vecs_temp[y[i]]
    labels[i] = labels_temp[y[i]]
end

-- generate trainset and testset
train_perc = 0.80 -- percentage of images in the train set
trainset_size = torch.round((#feat_vecs)[1] * train_perc)
trainset = feat_vecs[{{1, trainset_size}}]
train_labels = labels[{{1, trainset_size}}]
testset = feat_vecs[{{trainset_size+1, (#feat_vecs)[1]}}]
test_labels = labels[{{trainset_size+1, (#feat_vecs)[1]}}]

-- put the model in evalaute mode
BC_model:evaluate()

crit = nn.BCECriterion()
eval_crit = crit
lr = 2
attr_lr = 0.5
batch_size = 512
max_train_iter = 10000
test_interval = 50
test_iter = 1000
lr_stepsize = 100
gamma = 1
attr_gamma = 1
wd = 0
snapshot_interval = 100
snapshot_prefix = './'
snapshot = false
-- TO DO : Add weight decay

-- Start training
outfile = io.open('train_C.out', 'a')
outfile:write('Training with snapshotting ')
if snapshot then
    outfile:write('enabled... \n')
else
    outfile:write('disabled... \n')
end
outfile:close()


-- local method = 'xavier';
-- C_model.modules[2] = require('weight-init')(C_model.modules[2], method)
-- C_model.modules[6] = require('weight-init')(C_model.modules[6], method)

-- C_model.modules[2]:reset(0.01);
-- C_model.modules[6]:reset(0.01);
-- C_model.modules[10].modules[1].weight:mul(0.3)
-- C_model.modules[10].modules[3].weight:mul(0.3)
-- C_model.modules[10].modules[1].bias:mul(0.3)
-- C_model.modules[10].modules[3].bias:mul(0.09)

for i = 1, max_train_iter do

    -- initial testing
    if i == 1 then
        evalPerf(BC_model, eval_crit, testset, test_labels, test_iter)
    end

    BC_model:zeroGradParameters()
    local batch_loss = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(214325)

    local train_pred_err = 0
    for j = 1, batch_size do
        example = getCtrainExample(trainset, train_labels)
        input = example[1]
        target = example[2]
        local loss = 0
        local pred_err = 0
        loss, pred_err = accumulate(BC_model, {input[1], input[2], input[3], input[4]}, target, crit, eval_crit, batch_size, wd)
        batch_loss = batch_loss + loss
        train_pred_err = train_pred_err + pred_err;
        -- print(C_model.modules[9].output)
        -- print('loss =', loss)
        -- print(target)
    end

    -- update parameters for only a few layers in C
    -- C_model.modules[2]:updateParameters(attr_lr)
    -- C_model.modules[6]:updateParameters(attr_lr)
    -- C_model.modules[10]:updateParameters(lr)

    local grad_norm = torch.norm(C_model.modules[2].modules[3].gradWeight)

    outfile = io.open("train_C.out", "a")
    outfile:write('iter ', i, ', lr: ', lr, ', attr_lr: ', attr_lr)
    outfile:write(', batch_loss: ', batch_loss/batch_size, ', train_err: ', train_pred_err/batch_size)
    outfile:write(', grad_norm: ', grad_norm, '\n')
    outfile:close()

    if i % test_interval == 0 then
        evalPerf(BC_model, eval_crit, testset, test_labels, test_iter)        
    end

    if i % lr_stepsize == 0 then
        lr = lr * gamma;
        attr_lr = attr_lr * attr_gamma;
    end

    if snapshot and (i % snapshot_interval == 0) then

        outfile = io.open("train_C.out", "a")

        outfile:write('Snapshotting C_model... ')
        snapshot_filename_C = snapshot_prefix .. 'C_model__' .. tostring(i) .. '.t7'
        C_model:clearState()
        torch.save(snapshot_filename_C, C_model)
        outfile:write('done\n')

        outfile:close()

    end

end