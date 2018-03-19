require 'nngraph'

-- function to get an example for evaluating A
function getExample_A(set)

    -- randomly select two images from the set
    local y = torch.randperm((#set)[1])
    local input_temp = {}
    input_temp[1] = set[y[1]]
    input_temp[2] = set[y[2]]

    local input = torch.Tensor(4096*2)
    for i = 0, 1 do
        input[{{4096*i+1, 4096*(i+1)}}] = input_temp[i+1]:clone()
    end

    return input, y[1], y[2]
end


feat_vecs_test = torch.load('feat_vecs_test.t7')
probs_test_gt = torch.load('probs_test_gt.t7')
A_model = torch.load('A_model__900_init.t7')

num_eval = 2
for i = 1, num_eval do
	input, y1, y2 = getExample_A(feat_vecs_test)
	print(input:size())
	print(y1)
	print(y2)

	

end