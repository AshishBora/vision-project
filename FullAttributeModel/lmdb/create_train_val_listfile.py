import glob
import numpy as np

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


# randomly permute the data
perm = np.random.permutation(len(imPaths))
imPaths = [imPaths[i] for i in perm]
attrs = [attrs[i] for i in perm]

# split into train and val
train_size = int(len(imPaths) * 0.8)
imPaths_train = imPaths[:train_size]
attrs_train = attrs[:train_size]

imPaths_val = imPaths[train_size:]
attrs_val = attrs[train_size:]

print('train_size = ', len(imPaths_train))
print('val_size = ', len(attrs_val))

with open('train_listfile.txt', 'wb') as f:
	for imPath, attr in zip(imPaths_train, attrs_train):
		f.write(imPath[len(base_path):] + ' ' + str(attr) + '\n')


with open('val_listfile.txt', 'wb') as f:
    for imPath, attr in zip(imPaths_val, attrs_val):
        f.write(imPath[len(base_path):] + ' ' + str(attr) + '\n')
