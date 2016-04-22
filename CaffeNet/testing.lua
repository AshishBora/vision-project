require 'cudnn';
require 'cunn';
require 'image';
require 'loadcaffe';

-- get necessary functions
dofile('preproc.lua')
dofile('string_split.lua')
dofile('getImPaths.lua')

function testExample(model, input, targetValue)
	output = model:forward(input:cuda())
	max, indices = torch.max(output, 1)
	if(indices[1] == targetValue) then
		return 1
	else
		return 0
	end
end


-- load the model
model = torch.load('caffenet.t7')
model:evaluate()
model:cuda()


-- specify list file path and image directory path
list_file_path = '/work/04001/ashishb/maverick/data/listfiles/train_listfile_10.txt'
base_path = '/work/04001/ashishb/maverick/data/'


-- test on the images
no_of_correct_examples = 0
total_no_of_examples = 0
imPaths = getImPaths(list_file_path)
for label, v in pairs(imPaths) do
	for k, im_Path in pairs(v) do
		local total_path = base_path .. im_Path
		input = preprocess(total_path)
		no_of_correct_examples = no_of_correct_examples + testExample(model, input, label)
		total_no_of_examples = total_no_of_examples + 1
	end
end


-- print accuracy
print(no_of_correct_examples / total_no_of_examples)