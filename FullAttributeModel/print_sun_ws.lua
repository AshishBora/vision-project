require 'cudnn'
require 'torch'
require 'cunn';

sun_ws = torch.load('sun_ws.t7')
-- cudnn.convert(sun_ws, cudnn)

print(sun_ws)