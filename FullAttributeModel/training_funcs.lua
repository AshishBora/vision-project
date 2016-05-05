require 'nngraph';
require 'torch';



function create_BC(B_model, C_model)
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
function accumulate(model, inputs, targets, crit, wd)
    local probs = model:forward(inputs)
    local loss = crit:forward(probs, targets)
    local gradCriterion = crit:backward(probs, targets)
    -- model:backward(inputs, gradCriterion, 1/((#inputs)[1]))
    model:backward(inputs, gradCriterion)
    do_weight_decay(model, wd)
    local pred_err = get_total_pred_err(probs, targets)
    -- print('prob = ', prob)
    return loss, pred_err
end


-- function to evalaute the model
function evalPerf(model, crit, get_example, reader, iter, attrs, num_im)

    outfile = io.open("train_C.out", "a")
    outfile:write('Testing... \n')

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(3489208)
    inputs, targets = nextBatch(get_example, reader, iter, attrs, num_im)

    model:evalaute()
    local probs = model:forward(inputs)
    model:training()

    local test_loss = crit:forward(probs, targets)
    local test_pred_err = get_total_pred_err(probs, targets)

    outfile:write('test_loss = ', test_loss, ', ')
    outfile:write('test_pred_err = ', test_pred_err, '\n')
    outfile:close()

end



function nextBatch(get_example, reader, batch_size, attrs, num_im)
    local inputs = torch.Tensor(batch_size, 12330);
    local targets = torch.Tensor(batch_size);
    local i = 0;
    for i = 1, batch_size do
        inputs[i], targets[i] = get_example(reader, attrs, num_im)
    end
    inputs:double();
    targets:double();
    return inputs, targets
end