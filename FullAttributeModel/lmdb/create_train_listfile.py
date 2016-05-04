import glob

project_dir = '/Users/ashish/vision-project/'
base_path = project_dir + 'data/SUN/SUN_WS/training/'

attr_names = ['vegetation',
    'shrubbery',
    'foliage',
    'leaves',
    'shingles',
    'concrete',
    'metal',
    'paper',
    'wood',
    'vinyl',
    'rubber',
    'cloth',
    'sand',
    'rock',
    'dirt',
    'marble',
    'glass',
    'waves',
    'running_water',
    'still_water',
    'snow',
    'natural_light',
    'direct_sun',
    'electric',
    'aged',
    'glossy',
    'matte',
    'moist',
    'dry',
    'dirty',
    'rusty',
    'warm',
    'cold',
    'natural',
    'man_made',
    'open_area',
    'far_away_horizon',
    'rugged_scene',
    'symmetrical',
    'cluttered_space',
    'scary',
    'soothing']

imPaths = []
attrs = []
for i, attr_name in enumerate(attr_names):
	im_dir = base_path + attr_name + '/'
	im_paths_temp = glob.glob(im_dir + '*.jpg')
	attrs_temp = [i+1 for im in im_paths_temp]
	imPaths.extend(im_paths_temp)
	attrs.extend(attrs_temp)

print(len(imPaths))
print(len(attrs))

with open('train_listfile.txt', 'wb') as f:
	for imPath, attr in zip(imPaths, attrs):
		f.write(imPath[len(base_path):] + ' ' + str(attr) + '\n')

