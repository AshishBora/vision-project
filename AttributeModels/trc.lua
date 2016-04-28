
require 'nngraph';
require 'torch';

function createFullModel(B_model, C_model)
    local input = nn.Identity()();

    local image_feat1 = nn.Narrow(2, 1, 4096)(input);
    local image_feat2 = nn.Narrow(2, 4097, 4096)(input);
    local image_feat3 = nn.Narrow(2, 8193, 4096)(input);
    local question = nn.Narrow(2, 12289, 42)(input);
    
    

    local confidence = B_model({question, image_feat3});
    local scores = C_model({image_feat1, image_feat2, question, confidence});
    
    nngraph.annotateNodes();
    return nn.gModule({input}, {scores});
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
    
    local input = torch.cat({im_feat[1], im_feat[2], im_feat[3], ques})
--     print(#input)
    
    return {input, target}
end

function do_weight_decay(model, wd)
    lin_modules = model:findModules('nn.Linear');
    local i = 0;
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

function get_total_pred_err(probs, targets)
    local total_pred_err = 0;
    local i = 0;
    for i = 1, (#probs)[1] do
        total_pred_err = total_pred_err + get_pred_err(probs[i], targets[i]);
    end
    return total_pred_err/(#probs)[1];
end

-- Use a typical generic gradient update function
function accumulate(model, inputs, targets, criterion, eval_criterion,  wd)
    local probs = model:forward(inputs)
    local loss = criterion:forward(probs, targets)
    local gradCriterion = criterion:backward(probs, targets)
    model:backward(inputs, gradCriterion, 1/(#inputs)[1])
    do_weight_decay(model, wd)
    local pred_err = get_total_pred_err(probs, targets)
    -- print('prob = ', prob)
    return loss, pred_err
end


-- function to evalaute the model
function evalPerf(model, criterion, set, labels, iter)

    outfile = io.open("train_C.out", "a")
    outfile:write('Testing... ')


    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(3489208)
    inputs, targets = nextBatch(set, labels, iter);
--     model:forward(inputs[1])
--     print 'ttttt1\n'
--     model:forward(inputs[2])
--     print 'ttttt2\n'
--     model:forward(inputs[3])
--     print 'ttttt3\n'
   
    outfile:write('Fetched Batch');

--     local m1 = nn.Narrow(1, 1, 4096);
--     local m2 = nn.Narrow(1, 4097, 4096);
--     local m3 = nn.Narrow(1, 8193, 4096);
--     local m4 = nn.Narrow(1, 12289, 42);
--     m1:forward(inputs[1]);
--     m2:forward(inputs[1]);
--     m3:forward(inputs[1]);
--     m4:forward(inputs[1]);
--    model:forward(torch.Tensor(12330, 10));
    print('done')
    
     local probs = model:forward(inputs)
     local test_loss = criterion:forward(probs, targets)
     local test_pred_err = get_total_pred_err(probs, targets)

--     outfile:write('average test_loss = ', test_loss, ', ')
--     outfile:write('average test_pred_err = ', test_pred_err, '\n')
    outfile:close()
end


function nextBatch(trainset, train_labels, batchSize)
    local inputs = torch.Tensor(batchSize, 12330);
    local targets = torch.Tensor(batchSize);
    local i = 0;
    for i = 1, batchSize do
        example = getCtrainExample(trainset, train_labels);
        inputs[i] = example[1];
        targets[i] = example[2];
    end
    inputs:double();
    targets:double();
    return inputs, targets
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
C_model = torch.load('C_model.t7')

BC_model = createFullModel(B_model, C_model)
--graph.dot(BC_model.fg, 'BCM')

-- convert to double
BC_model:double()

outfile:write('done\n')
outfile:close()


-- read preprocessed feature vectors and labels
feat_vecs = torch.load('feat_vecs.t7')
labels = torch.load('labels.t7')

-- TO DO : random shuffle of data
-- y = torch.randperm(#feat_vecs)

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
attr_lr = 1
batch_size = 512
max_train_iter = 10000
test_interval = 50
test_iter = 1000
lr_stepsize = 100
gamma = 0.7
attr_gamma = 0.5
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

print(C_model.modules) 
C_model_old = torch.load('C_model__1500_init.t7')
C_model.modules[13].modules[1].weight = C_model_old.modules[10].modules[1].weight
C_model.modules[13].modules[3].weight = C_model_old.modules[10].modules[3].weight
-- local method = 'xavier';
-- C_model.modules[2] = require('weight-init')(C_model.modules[2], method)
-- C_model.modules[6] = require('weight-init')(C_model.modules[6], method)

-- C_model.modules[2]:reset(0.01);
-- C_model.modules[6]:reset(0.01);
-- C_model.modules[10].modules[1].weight:mul(0.3)
-- C_model.modules[10].modules[3].weight:mul(0.3)

for i = 1, max_train_iter do

    -- initial testing

    if i == 1 then
        evalPerf(BC_model, eval_crit, testset, test_labels, test_iter)
    end
    BC_model:zeroGradParameters()
    print('i ', i, '\n');
    local batch_loss = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(214325)

    inputs, targets = nextBatch(trainset, train_labels, batch_size);
    batch_loss, train_pred_err = accumulate(BC_model, inputs, targets, crit, eval_crit,  wd);

    -- update parameters for only a few layers in C
--     C_model.modules[2]:updateParameters(attr_lr)
--     C_model.modules[6]:updateParameters(attr_lr)
--     C_model.modules[13]:updateParameters(lr)
    
    BC_model:clearState();

    outfile = io.open("train_C.out", "a")
    outfile:write('Iteration no. ', i, ', lr = ', lr, ', attr_lr = ', attr_lr, ', batch_loss = ', batch_loss, ', train_err = ', train_pred_err, '\n')
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
