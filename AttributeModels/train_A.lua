require 'nngraph';

function create_Full(A_model, B_model, C_model)
    local image_feat1 = nn.Identity()();
    local image_feat2 = nn.Identity()();
    local image_feat3 = nn.Identity()();

    local question = A_model({image_feat1, image_feat2});
    local confidence = B_model({question, image_feat3});
    local prob = C_model({image_feat1, image_feat2, question, confidence});

    nngraph.annotateNodes();
    return nn.gModule({image_feat1, image_feat2, image_feat3}, {prob});
end

-- function to get an example for training/testing A
function getExample_A(set)

    -- randomly select two images from the set
    local y = torch.randperm((#set)[1])
    local input = {}
    input[1] = set[y[1]]
    input[2] = set[y[2]]

    -- randomly select one of those to be given to model B
    local label = torch.bernoulli() + 1
    input[3] = input[label]:clone()

    -- target is probability of label = 2
    local target = label - 1
   
    return {input, target}
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

-- Generic gradient update function
function accumulate(model, input, target, criterion, batch_size)
    local prob = model:forward(input)
    local loss = criterion:forward(prob, torch.Tensor{target})
    local gradCriterion = criterion:backward(prob, torch.Tensor{target})
    model:backward(input, gradCriterion, 1/batch_size)
    local pred_err = get_pred_err(prob, target)
    return loss, pred_err
end


-- function to evaluate the model
function evalPerf(model, criterion, set, labels, iter)

    outfile = io.open("train_A.out", "a")
    outfile:write('Testing... ')
    outfile:close()

    local test_loss = 0
    local test_pred_err = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(3489208)

    for j = 1, iter do
        example = getExample_A(set)
        input = example[1]
        target = example[2]
        local prob = model:forward(input)
        local samp_loss = criterion:forward(prob, torch.Tensor{target})
        local pred_err = get_pred_err(prob, target)
        test_pred_err = test_pred_err + pred_err
        test_loss = test_loss + samp_loss
    end

    outfile = io.open("train_A.out", "a")
    outfile:write('test_loss = ', test_loss/iter, ', ')
    outfile:write('test_pred_err = ', test_pred_err/iter, '\n')
    outfile:close()
end



-- Laod A, B and C models
outfile = io.open("train_A.out", "w")
outfile:write('Loading pretrained model... ')
A_model = torch.load('A_model.t7')
B_model = torch.load('B_model_nn.t7')
C_model = torch.load('C_model__1500.t7')
outfile:write('done\n')
outfile:close()

-- Create Full model
outfile = io.open("train_A.out", "a")
outfile:write('Creating Full model... ')
Full_model = create_Full(A_model, B_model, C_model)
Full_model:double() -- convert to double
Full_model:evaluate() -- put in evalaute mode
outfile:write('done\n')
outfile:close()


-- read preprocessed feature vectors and labels
outfile = io.open("train_A.out", "a")
outfile:write('Reading features and labels... ')
feat_vecs = torch.load('feat_vecs.t7')
labels = torch.load('labels.t7')
outfile:write('done\n')
outfile:close()

-- TO DO : random shuffle of data
-- y = torch.randperm(#feat_vecs)

-- generate trainset and testset
train_perc = 0.80 -- percentage of images in the train set
trainset_size = torch.round((#feat_vecs)[1] * train_perc)
trainset = feat_vecs[{{1, trainset_size}}]
train_labels = labels[{{1, trainset_size}}]
testset = feat_vecs[{{trainset_size+1, (#feat_vecs)[1]}}]
test_labels = labels[{{trainset_size+1, (#feat_vecs)[1]}}]


---------------- Training -----------------------------

-- set training hyperparameters
crit = nn.BCECriterion()
-- lr = 0.001 : diverges
lr = 0.0001
batch_size = 512
max_train_iter = 10000
test_interval = 50
test_iter = 1000
lr_stepsize = 100
gamma = 0.7
snapshot_interval = 100
snapshot_prefix = './'
snapshot = true -- set to false by default to avoid overwriting
-- TO DO : Add weight decay

-- Start training
outfile = io.open("train_A.out", "a")
outfile:write('Training with snapshotting ')
if snapshot then
    outfile:write('enabled... \n')
else
    outfile:write('disabled... \n')
end
outfile:close()

for i = 1, max_train_iter do

    -- initial testing
    if i == 1 then
        evalPerf(Full_model, crit, testset, test_labels, test_iter)
    end

    Full_model:zeroGradParameters()
    local batch_loss = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(214325)

    local train_pred_err = 0
    for j = 1, batch_size do
        example = getExample_A(trainset)
        input = example[1]
        target = example[2]
        local loss = 0
        local pred_err = 0
        loss, pred_err = accumulate(Full_model, input, target, crit, batch_size)
        batch_loss = batch_loss + loss
        train_pred_err = train_pred_err + pred_err;
    end

    -- update parameters only for A
    A_model:updateParameters(lr)

    outfile = io.open("train_A.out", "a")
    outfile:write('Iteration no. ', i, ', lr = ', lr, ', batch_loss = ', batch_loss/batch_size, ', train_err = ', train_pred_err/batch_size, '\n')
    outfile:close()

    if i % test_interval == 0 then
        evalPerf(Full_model, crit, testset, test_labels, test_iter)        
    end

    if i % lr_stepsize == 0 then
        lr = lr * gamma
    end

    if snapshot and (i % snapshot_interval == 0) then

        outfile = io.open("train_A.out", "a")

        outfile:write('Snapshotting A_model... ')
        snapshot_filename_A = snapshot_prefix .. 'A_model__' .. tostring(i) .. '.t7'
        torch.save(snapshot_filename_A, A_model)
        outfile:write('done\n')

        outfile:close()

    end

end
