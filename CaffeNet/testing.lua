require 'cudnn';
require 'cunn';
require 'image';
require 'loadcaffe';

function testExample(model, input, targetValue)
	output=model:forward(input:cuda())
	max, indices = torch.max(output,1)
	if(indices[1] == (targetValue+1)) then
		return 1
	else
		return 0
	end
end


dofile('preproc.lua')
dofile('string_split.lua')
-- load the model                                                             
model = torch.load('caffenet.t7')
model:evaluate()
model:cuda()


list_file_path = '/work/04001/ashishb/maverick/data/listfiles/train_listfile_100.txt'
base_path = '/work/04001/ashishb/maverick/data/'
-- file = io.open(list_file_path, 'r')

no_of_correct_examples=0
total_no_of_examples=0
for line in io.lines(list_file_path) do

	words = line:split(' ')
	im_Path, label=words[1], words[2]
	label=tonumber(label)

	local total_path=base_path .. im_Path
	input = preprocess(total_path)
	
	no_of_correct_examples = no_of_correct_examples + testExample(model, input, label)
	total_no_of_examples = total_no_of_examples + 1
	
end

print(no_of_correct_examples/total_no_of_examples)


