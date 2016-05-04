import glob

project_dir = '/Users/ashish/vision-project/'
base_path = project_dir + 'data/SUN/SUN_WS/training/'

attrs = ['vegetation',
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
labels = []
for i, attr in enumerate(attrs):
	im_dir = base_path + attr + '/'
	im_paths_temp = glob.glob(im_dir + '*.jpg')
	labels_temp = [i+1 for im in im_paths_temp]
	imPaths.extend(im_paths_temp)
	labels.extend(labels_temp)

print(len(imPaths))
print(len(labels))

with open('train_listfile.txt', 'wb') as f:
	for imPath, label in zip(imPaths, labels):
		f.write(imPath[len(base_path):] + ' ' + str(label) + '\n')

