require 'torch'
require 'torchx'

function getCrossEnt(prob, target)
	local log_prob = torch.log(prob)
	local log_reverse_prob = torch.log(1 - prob)
	local CrossEnt = -torch.mean(torch.cmul(target, log_prob) + torch.cmul(1 - target, log_reverse_prob))
	return CrossEnt
end

function getTopK(input, k)
	if k == 0 then
		return {}
	end
	output, indices = torch.topk(input, k, 1, true)
	return indices
end
-- A = torch.Tensor{1, 2, 5, 3, 4, 2, 1, 3, 4}
-- output, indices = getTopK(A, 3)
-- print(output)
-- print(indices)

function getNonZero(input)
	local nonZero = torch.gt(input, 0)
	local indices = torch.find(nonZero, 1)
	return indices
end
-- A = torch.Tensor{1, 0, 5, -3, 4, -2, -1, 0, 4}
-- indices = getNonZero(A)
-- for k, v in pairs(indices) do
--    print(v)
-- end
-- print(#indices)

function getIntersection(P, T)
	local intersection = {}
	P = torch.sort(P)
	T = torch.sort(T)
	local it = 1
	local ip = 1
	while (ip <= (#P)[1]) and (it <= (#T)[1]) do
		p = P[ip]
		t = T[it]
		if t == p then
			table.insert(intersection, t)
			it = it + 1
			ip = ip + 1
		elseif t < p then
			it = it + 1
		else
			ip = ip + 1
		end
	end
	return intersection
end
-- T = torch.Tensor{6, 2, 5, 1}
-- P = torch.Tensor{1, 5, 7, 8}
-- intersection = getIntersection(T, P)
-- for k, v in pairs(intersection) do
--    print(v)
-- end
-- print(#intersection)

function getPrecision(prob, target)

	local precision = 0
	local num_im = target:size()[1]
	print('Total number of test images =', num_im)

	for i = 1, num_im do
		trgt = target[i]
		T = getNonZero(trgt)
		k = #T
		-- print(k)
		prb = prob[i]
		P = getTopK(prb, k)

		T = torch.Tensor(T)
		-- print('T = ')
		-- print(T)
		-- print('P = ')
		-- print(P)

		intersection = getIntersection(T, P)
		prec = (#intersection) / k
		precision = precision + prec
	end

	precision = precision / num_im
	return precision
end


target = torch.load('probs_test_gt.t7')
prob = torch.load('probs_test.t7')

-- print(probs_gt:size())
-- print(probs:size())

-- print(torch.max(probs_gt))
-- print(torch.min(probs_gt))

-- print(torch.max(probs))
-- print(torch.min(probs))

-- crossEnt = getCrossEnt(prob, target)
-- print(crossEnt)

precision = getPrecision(prob, target)
print('Average precision = ', precision)