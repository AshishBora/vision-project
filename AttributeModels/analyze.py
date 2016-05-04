import matplotlib.pyplot as plt
import numpy as np

lines = []
with open('./train_ABC.out', 'r') as f:
	line = f.readline()
	while line != '':
		lines.append(line)
		line = f.readline()
lines = lines[3:]



###################################################
batch_loss = []
for line in lines:
	# print line
	if line.find('batch_loss') != -1:
		batch_loss.append(float(line.split(' ')[7][:-1]))
batch_loss_plot, = plt.plot(batch_loss)


###################################################
train_err = []
for line in lines:
	# print line
	if line.find('train_err') != -1:
		train_err.append(float(line.split(' ')[9][:-1]))
train_err_plot, = plt.plot(train_err)


###################################################
test_loss = []
for line in lines:
	# print line
	if line.find('test_loss') != -1:
		test_loss.append(float(line.split(' ')[3][:-1]))
test_loss_plot, = plt.plot(50 * np.arange(len(test_loss)), test_loss)


###################################################
test_pred_err = []
for line in lines:
	# print line
	if line.find('test_pred_err') != -1:
		test_pred_err.append(float(line.split(' ')[7][:-1]))
test_pred_err_plot, = plt.plot(50 * np.arange(len(test_pred_err)), test_pred_err)



plots = [batch_loss_plot,
		 train_err_plot,
		 test_loss_plot,
		 test_pred_err_plot]

legends = ['batch_loss',
		   'train_err',
		   'test_loss',
		   'test_err']

plt.legend(plots, legends)
plt.show()
