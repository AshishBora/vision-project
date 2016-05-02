require 'nngraph';
require 'torch';


function createFullModel(A_model, B_model, C_model)
    local input = nn.Identity()();

    local image_feat1 = nn.Narrow(2, 1, 4096)(input);
    local image_feat2 = nn.Narrow(2, 4097, 4096)(input);
    local image_feat3 = nn.Narrow(2, 8193, 4096)(input);

    local question = A_model({image_feat1, image_feat2});
    local confidence = B_model({question, image_feat3});
    local prob = C_model({image_feat1, image_feat2, question, confidence});

    nngraph.annotateNodes();
    return nn.gModule({input}, {prob});
end




-- function to get an example for training/testing A
function getExample_A(set)

    -- randomly select two images from the set
    local y = torch.randperm((#set)[1])
    local input_temp = {}
    input_temp[1] = set[y[1]]
    input_temp[2] = set[y[2]]

    -- randomly select one of those to be given to model B
    local label = torch.bernoulli() + 1
    input_temp[3] = input_temp[label]:clone()

    -- target is probability of label = 2
    local target = label - 1

    local input = torch.Tensor(4096*3)
    for i = 0, 2 do
        input[{{4096*i+1, 4096*(i+1)}}] = input_temp[i+1]:clone()
    end

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
    -- model:backward(inputs, gradCriterion, 1/((#inputs)[1]))
    model:backward(inputs, gradCriterion)
    do_weight_decay(model, wd)
    local pred_err = get_total_pred_err(probs, targets)
    -- print('prob = ', prob)
    return loss, pred_err
end


-- function to evaluate the model
function evalPerf(model, criterion, set, iter)

    outfile = io.open("train_A.out", "a")
    outfile:write('Testing... \n')

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(3489208)
    inputs, targets = nextBatch(set, iter);

    model:evaluate()
    local probs = model:forward(inputs)
    local test_loss = criterion:forward(probs, targets)
    model:training()

    local test_pred_err = get_total_pred_err(probs, targets)

    outfile:write('average test_loss = ', test_loss, ', ')
    outfile:write('average test_pred_err = ', test_pred_err, '\n')
    outfile:close()

end


function nextBatch(trainset, batchSize)
    local inputs = torch.Tensor(batchSize, 4096*3);
    local targets = torch.Tensor(batchSize);
    local i = 0;
    for i = 1, batchSize do
        example = getExample_A(trainset)
        inputs[i] = example[1];
        targets[i] = example[2];
    end
    inputs:double();
    targets:double();
    return inputs, targets
end


---------------- Read preprocessed feature vectors ------------------------
outfile = io.open("train_A.out", "a")
outfile:write('Preparing data... ')

feat_vecs = torch.load('feat_vecs.t7')
-- labels = torch.load('labels.t7')

-- randomly shuffle data
feat_vecs_temp = feat_vecs:clone()
-- labels_temp = labels:clone()
y = torch.randperm((#feat_vecs)[1])
for i = 1, ((#feat_vecs)[1]) do
    feat_vecs[i] = feat_vecs_temp[y[i]]
    -- labels[i] = labels_temp[y[i]]
end

-- garbage collection
feat_vecs_temp = nil
-- labels_temp = nil

-- generate trainset and testset
train_perc = 0.80 -- percentage of images to be taken in the train set
trainset_size = torch.round((#feat_vecs)[1] * train_perc)
trainset = feat_vecs[{{1, trainset_size}}]
-- train_labels = labels[{{1, trainset_size}}]
testset = feat_vecs[{{trainset_size+1, (#feat_vecs)[1]}}]
-- test_labels = labels[{{trainset_size+1, (#feat_vecs)[1]}}]

outfile:write('done\n')
outfile:close()



-------------------------  Create model --------------------------
outfile = io.open("train_A.out", "w")
outfile:write('Creating model... ')

A_model = torch.load('A_model.t7')
B_model = torch.load('B_model_nn.t7')
C_model = torch.load('C_model__10000.t7')

ABC_model = createFullModel(A_model, B_model, C_model)
-- graph.dot(ABC_model.fg, 'ABCM')
crit = nn.BCECriterion() -- define loss

outfile:write('done\n')
outfile:close()




----------------  Weight initialization  ------------------------

-- random initialization of A's fully connected layer
-- A_model.modules[2]:reset(0.01)
-- A_model.modules[5]:reset(0.01)

-- divide weights to get in trainable area
div_fact = 0.3
C_model.modules[13].modules[1].weight:mul(div_fact)
C_model.modules[13].modules[3].weight:mul(div_fact)
C_model.modules[13].modules[1].bias:mul(div_fact)
C_model.modules[13].modules[3].bias:mul(div_fact^2)




---------------- Define hyper parameters ------------------------
lr = 0.2
attr_lr = 0
batch_size = 512
max_train_iter = 10000
test_interval = 50
test_iter = 1000
lr_stepsize = 200
gamma = 1
attr_gamma = 1
wd = 0
snapshot_interval = 100
snapshot_prefix = './'
snapshot = false
B_model.modules[2].modules[2].p = 0.2 -- dropout rate
C_model.modules[2].modules[2].p = 0.2  -- dropout rate
C_model.modules[7].modules[2].p = 0.2  -- dropout rate

-- no dropout in A
A_model.modules[2].modules[2].p = 0
A_model.modules[5].modules[2].p = 0

----------------  Start training ------------------------

ABC_model:double() -- convert to double
ABC_model:training() -- enable all dropouts

outfile = io.open('train_A.out', 'a')
outfile:write('Training with snapshotting ')
if snapshot then
    outfile:write('enabled... \n')
else
    outfile:write('disabled... \n')
end
outfile:close()

for i = 1, max_train_iter do

    collectgarbage()

    -- initial testing
    if i == 1 then
        evalPerf(ABC_model, crit, testset, test_iter)
    end

    ABC_model:zeroGradParameters()
    local batch_loss = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(214325)

    inputs, targets = nextBatch(trainset, batch_size);
    batch_loss, train_pred_err = accumulate(ABC_model, inputs, targets, crit, crit,  wd);

    -- update parameters for A
    -- C_model.modules[2]:updateParameters(attr_lr)
    -- C_model.modules[7]:updateParameters(attr_lr)
    -- C_model.modules[13]:updateParameters(lr)
    A_model.modules[8]:updateParameters(lr)

    ABC_model:clearState(); -- reduce memory usage

    local grad_norm = torch.norm(A_model.modules[8].gradWeight * lr)

    outfile = io.open("train_A.out", "a")
    outfile:write('iter ', i, ', lr: ', lr, ', attr_lr: ', attr_lr)
    outfile:write(', batch_loss: ', batch_loss, ', train_err: ', train_pred_err)
    outfile:write(', grad_norm: ', grad_norm, '\n')
    outfile:close()

    if i % test_interval == 0 then
        evalPerf(ABC_model, crit, testset, test_iter)        
    end

    if i % lr_stepsize == 0 then
        lr = lr * gamma;
        attr_lr = attr_lr * attr_gamma;
    end

    if snapshot and (i % snapshot_interval == 0) then

        outfile = io.open("train_A.out", "a")

        outfile:write('Snapshotting A_model... ')
        snapshot_filename_A = snapshot_prefix .. 'A_model__' .. tostring(i) .. '.t7'
        A_model:clearState()
        torch.save(snapshot_filename_C, C_model)
        outfile:write('done\n')

        outfile:close()

    end

end