project_dir = '/Users/ashish/vision-project/'
base_path = project_dir + 'data/SUN/SUN_WS/test/'

imPaths = []
for i in range(5618):
    im_path = base_path + 'sun_ws_test_' + str(i+1) + '.jpg'
    imPaths.append(im_path)

print(len(imPaths))

with open('test_listfile.txt', 'wb') as f:
	for imPath in imPaths:
		f.write(imPath[len(base_path):] + ' ' + str(0) + '\n')

