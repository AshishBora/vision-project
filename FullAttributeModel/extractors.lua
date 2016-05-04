-- functions to get feature and attribure extractors from the original sun_ws

require 'cudnn';
require 'nngraph';
require 'cunn';

function get_getFeat(sun_ws, mimick)
	local getFeat = nn.Sequential()
	for i = 1, 20 do
		getFeat:add(sun_ws.modules[i]:clone())
	end
	if mimick == false then
		getFeat:reset()
	end
	return getFeat
end


function get_getAttr(sun_ws, mimick)
	local getAttr = nn.Sequential()
	for i = 21, 23 do
		getAttr:add(sun_ws.modules[i]:clone())
	end
	getAttr:add(nn.Sigmoid())
	if mimick == false then
		getAttr:reset()
	end
	return getAttr
end

function get_predictor(B_model)
	local predictor = nn.Sequential()
	for i = 2, 3 do
		predictor:add(B_model.modules[i]:clone())
	end
	return predictor
end