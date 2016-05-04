-- functions to get feature and attribure extractors from the original model

require 'cudnn';
require 'nngraph';
require 'cunn';

function get_getFeat(model)
	local getFeat = nn.Sequential()
	for i = 1, 20 do
		getFeat:add(model.modules[i]:clone())
	end
	getFeat:reset()
	return getFeat
end


function get_getAttr(model)
	local getAttr = nn.Sequential()
	for i = 21, 23 do
		getAttr:add(model.modules[i]:clone())
	end
	getAttr:add(nn.Sigmoid())
	getAttr:reset()
	return getAttr
end