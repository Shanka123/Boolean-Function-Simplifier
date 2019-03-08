import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
import numpy as np
from torchvision import transforms,datasets, models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import torchvision
import pickle
from sklearn import model_selection
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import copy
import time
from sklearn.metrics import confusion_matrix

input_arr = np.load('kmap_all_input_data_new.npy')
target_arr = np.load('kmap_all_output_data_new.npy')

class K_map(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, input_arr, target_arr):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
#         self.landmarks_frame = pd.read_csv(csv_file)
		
		self.input_arr = input_arr
		self.target_arr = target_arr

	def __len__(self):
		return self.target_arr.shape[0]

	def __getitem__(self, idx):
		inp = self.input_arr[idx,:,:]
		inp = inp.reshape((1, self.input_arr.shape[1], self.input_arr.shape[2]))
		target = self.target_arr[idx,:]
		inp = torch.from_numpy(inp)
		target = torch.from_numpy(target)

		return inp, target

num_ones = np.sum(target_arr, axis = 1)
train_input_arr, temp_inp, train_target_arr, temp_target = model_selection.train_test_split(input_arr, target_arr, test_size = 0.3, random_state = 1, stratify = num_ones)
num_ones_temp = np.sum(temp_target, axis = 1)
val_input_arr, test_input_arr, val_target_arr, test_target_arr = model_selection.train_test_split(temp_inp, temp_target, test_size = 0.33, random_state = 1, stratify = num_ones_temp)

train_data = K_map(input_arr = train_input_arr, target_arr = train_target_arr)
val_data = K_map(input_arr = val_input_arr, target_arr = val_target_arr)
test_data = K_map(input_arr = test_input_arr, target_arr = test_target_arr)

BatchSize = 64
trainLoader = torch.utils.data.DataLoader(train_data, batch_size=BatchSize, shuffle=True, num_workers=4)
valLoader = torch.utils.data.DataLoader(val_data, batch_size=BatchSize, shuffle=True, num_workers=4)
testLoader = torch.utils.data.DataLoader(test_data, batch_size=BatchSize, shuffle=False, num_workers=4)

print('No. of samples in train set: '+str(len(trainLoader.dataset)))
print('No. of samples in validation set: '+str(len(valLoader.dataset)))
print('No. of samples in test set: '+str(len(testLoader.dataset)))

use_gpu = torch.cuda.is_available()
if use_gpu:
    print('GPU is available!')

class arch(nn.Module):
	def __init__(self):
		super(arch, self).__init__()
		
		self.pad_1x2 = nn.ZeroPad2d((0, 1, 0, 0))
		self.pad_2x1 = nn.ZeroPad2d((0, 0, 0, 1))
		self.pad_2x2 = nn.ZeroPad2d((0, 1, 0, 1))
		self.pad_1x4 = nn.ZeroPad2d((1, 2, 0, 0))
		self.pad_4x1 = nn.ZeroPad2d((0, 0, 1, 2))
		self.pad_2x4 = nn.ZeroPad2d((1, 2, 0, 1))
		self.pad_4x2 = nn.ZeroPad2d((0, 1, 1, 2))
		self.pad_4x4 = nn.ZeroPad2d((1, 2, 1, 2))
		self.conv_1x2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=1)
		self.conv_2x1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2,1), stride=1)
		self.conv_2x2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2,2), stride=1)
		self.conv_2x4 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2,4), stride=1)
		self.conv_4x2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,2), stride=1)
		self.conv_4x4 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(4,4), stride=1)
		self.conv_4x1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,1), stride=1)
		self.conv_1x4 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,4), stride=1)
		self.conv_1x1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,1), stride=1)

		self.fc = nn.Sequential(nn.Linear(in_features = 256*4*4 + 96*4*4, out_features = 80, bias = True))
	def forward(self, x):
		# print(x.size())
		# exit()
		x1 = self.pad_1x2(x)
		x1 = self.conv_1x2(x1)
		x1 = F.relu(x1)
		x2 = self.pad_2x1(x)
		x2 = self.conv_2x1(x2)
		x2 = F.relu(x2)
		x3 = self.pad_2x2(x)
		x3 = self.conv_2x2(x3)
		x3 = F.relu(x3)
		x4 = self.pad_2x4(x)
		x4 = self.conv_2x4(x4)
		x4 = F.relu(x4)
		x5 = self.pad_4x2(x)
		x5 = self.conv_4x2(x5)
		x5 = F.relu(x5)
		x6 = self.pad_4x4(x)
		x6 = self.conv_4x4(x6)
		x6 = F.relu(x6)
		x7 = self.pad_1x4(x)
		x7 = self.conv_1x4(x7)
		x7 = F.relu(x7)
		x8 = self.pad_4x1(x)
		x8 = self.conv_4x1(x8)
		x8 = F.relu(x8)
		
		x9 = self.conv_1x1(x)
		x9 = F.relu(x9)
		x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9), 1)
		x = x.view(-1,256*4*4 + 96*4*4)
		x = self.fc(x)
		return(x)

net = arch()
if use_gpu:
	net = net.double().cuda()
print(net)

criterion = nn.MultiLabelSoftMarginLoss()
opt = optim.Adam(net.parameters(), lr = 5e-4)

print(trainLoader.batch_size)


# In[16]:

def compare(pred_tensor, target):
	temp1 = (pred_tensor.view(-1)>0).nonzero()
	if temp1.size()[0] == 0:
		return(0)
	temp1 = temp1.view(-1)
	temp2 = (target.view(-1).int() == 1).nonzero().view(-1)
	temp1, _ = torch.sort(temp1)
	temp2, _ = torch.sort(temp2)
	temp1=temp1.int()
	temp2=temp2.int()

	if temp1.size()==temp2.size():
		# print('hello')
		# print(temp1, temp2)
		if (temp1==temp2).sum().item() == temp1.size()[0]:

			return(1)
		else:
			# print(temp1, temp2)
			return(0)
	else:
		return(0)
import warnings
warnings.filterwarnings('ignore')
trainLoss = [] # List for saving main loss per epoch
trainAcc = [] # List for saving training accuracy per epoch
valLoss = [] # List for saving testing loss per epoc
valAcc = []
testAcc = []
testLoss = []

total_train_samples = float(len(trainLoader.dataset))
total_val_samples = float(len(valLoader.dataset))

iterations = 800
min_val_loss = float(sys.maxsize)
max_test_acc = 0
best_net = net
start = time.time()
for epoch in range(iterations):
	runningLoss = 0.0 
	avg_Loss = 0.0
	running_correct = 0

	net.train(True)

	for idx, data in enumerate(trainLoader):
		inputs,labels = data
		# print(inputs.size())
		# Wrap them in Variable
		if use_gpu:
			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
			predicted = net(inputs)
			opt.zero_grad()
			loss = criterion(predicted, labels)
			loss.backward()
			opt.step()
			runningLoss += loss.data[0] 
		# for i in range(predicted.size()[0]):
		# 	pred_top2 = torch.topk(predicted[i].data.cpu(), 2)[1]
		# 	labels_top2 = torch.topk(labels[i].cpu(), 2)[1]
		# 	# print(pred_top2)
		# 	# print(labels_top2)
		# 	if (pred_top2 == labels_top2).sum() == 2 or (pred_top2 == torch.index_select(labels_top2,0,torch.Tensor([1,0]).long())).sum() == 2:
		# 		running_correct += 1
		# print('done')
		for i in range(predicted.size()[0]):
			running_correct+=compare(predicted[i], labels[i])
		# print(running_correct)
	avg_train_acc = float(running_correct)/len(trainLoader.dataset)
	trainAcc.append(avg_train_acc)
	avg_Loss = runningLoss/float(idx + 1)		
	trainLoss.append(avg_Loss)


	net.train(False)

	runningLoss = 0.0
	running_correct = 0

	for idx,data in enumerate(valLoader):
		inputs,labels = data
		# Wrap them in Variable
		if use_gpu:
			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
		predicted = net(inputs)
		loss = criterion(predicted, labels)
		runningLoss += loss.data[0]
		for i in range(predicted.size()[0]):
			# print(predicted[i], labels[i])
			running_correct+=compare(predicted[i], labels[i])
		# for i in range(predicted.size()[0]):
		# 	pred_top2 = torch.topk(predicted[i].data.cpu(), 2)[1]
		# 	labels_top2 = torch.topk(labels[i].cpu(), 2)[1]
		# 	# print(pred_top2)
		# 	# print(labels_top2)
		# 	if (pred_top2 == labels_top2).sum() == 2 or (pred_top2 == torch.index_select(labels_top2,0,torch.Tensor([1,0]).long())).sum() == 2:
		# 		running_correct += 1

	avg_val_acc = float(running_correct)/len(valLoader.dataset)
	valAcc.append(avg_val_acc)
	avg_val_loss = runningLoss/float(idx + 1)
	valLoss.append(avg_val_loss)
	if avg_val_loss < min_val_loss:
		min_val_loss = avg_val_loss
		best_net = net
	print('Iteration: {:.0f} /{:.0f} Model ; time: {:.3f} secs'.format(epoch + 1,iterations, (time.time() - start)))
	print('Training Loss: {:.6f} '.format(avg_Loss))
	print('Validation Loss: {:.6f} '.format(avg_val_loss))
	print('Training Acc: {:.3f} '.format(avg_train_acc*100))
	print('Validation Acc: {:.3f} '.format(avg_val_acc*100))

	runningLoss = 0.0 
	avg_test_loss = 0.0
	running_correct = 0


	for idx, data in enumerate(testLoader):
		inputs,labels = data
		# print(inputs.size())
		# Wrap them in Variable
		if use_gpu:
			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
		# test_time_1 = time.time()
		predicted = net(inputs)
		# print((time.time() - test_time_1)/BatchSize)
		loss = criterion(predicted, labels)
		runningLoss += loss.data[0]
		for i in range(predicted.size()[0]):
			running_correct+=compare(predicted[i], labels[i])
		# for i in range(predicted.size()[0]):
		# 	print(predicted[i])
		# 	print(labels[i])
		# for i in range(predicted.size()[0]):
		# 	pred_top2 = torch.topk(predicted[i].data.cpu(), 2)[1]
		# 	labels_top2 = torch.topk(labels[i].cpu(), 2)[1]
		# 	print(pred_top2)
		# 	print(labels_top2)
		# 	if (pred_top2 == labels_top2).sum() == 2 or (pred_top2 == torch.index_select(labels_top2,0,torch.Tensor([1,0]).long())).sum() == 2:
		# 		running_correct += 1

	test_acc = float(running_correct)/len(testLoader.dataset)
	torch.save(net.state_dict(), 'running_new_prop1_model.pt')
	with open('new_prop1_present_epoch.txt','w') as f:
		f.write(str(epoch + 1))
	if test_acc > max_test_acc:
		max_test_acc = test_acc
		temp_epoch = epoch + 1
		torch.save(net.state_dict(), 'best_new_prop1_model.pt')
	avg_test_loss = runningLoss/float(idx + 1)
	testLoss.append(avg_test_loss)
	testAcc.append(test_acc)
	print('-----------------Testing---------------------')
	print(avg_test_loss, test_acc * 100)
print('\n\n\n\n')
print(max_test_acc, temp_epoch)

net = arch()
net.load_state_dict(torch.load('best_new_prop1_model.pt', map_location='cpu'))
print('Model loaded !')
net = net.double().cuda()

target_list = []
pred_list = []
for idx, data in enumerate(testLoader):
	inputs,labels = data
	# print(inputs.size())
	# Wrap them in Variable
	if use_gpu:
		inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
	# test_time_1 = time.time()
	predicted = net(inputs)
	# print((time.time() - test_time_1)/BatchSize)
	loss = criterion(predicted, labels)
	runningLoss += loss.data[0]
	for i in range(predicted.size()[0]):
		target_list.append(np.array(labels[i].view(-1)).astype(int))
		pred_list.append(np.array((predicted[i].view(-1) > 0)).astype(int))

target_arr = np.array(target_list)
pred_arr = np.array(pred_list)

from sklearn.metrics import precision_recall_fscore_support
pr, rec,f_beta, _ = precision_recall_fscore_support(target_arr, pred_arr)

print('\n\n\n\n')
print(np.mean(pr))
print(np.mean(rec))
print(np.mean(f_beta))


np.save('all_prop1_new_test_acc.npy', np.array(testAcc))
np.save('all_prop1_new_val_acc.npy', np.array(valAcc))
np.save('all_prop1_new_train_acc.npy', np.array(trainAcc))

np.save('all_prop1_new_test_loss.npy', np.array(testLoss))
np.save('all_prop1_new_val_loss.npy', np.array(valLoss))
np.save('all_prop1_new_train_loss.npy', np.array(trainLoss))
# net.train(False)
# best_net.train(False)

# print('Using the last')

# runningLoss = 0.0 
# avg_test_loss = 0.0
# running_correct = 0


# for idx, data in enumerate(testLoader):
# 	inputs,labels = data
# 	# print(inputs.size())
# 	# Wrap them in Variable
# 	if use_gpu:
# 		inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
# 	predicted = net(inputs)
# 	loss = criterion(predicted, labels)
# 	runningLoss += loss.data[0]

# 	print(predicted.size())
# 	print(labels.size())
# 	for i in range(predicted.size()[0]):
# 		pred_top2 = torch.topk(predicted[i].data.cpu(), 2)[1]
# 		labels_top2 = torch.topk(labels[i].cpu(), 2)[1]
# 		print(pred_top2)
# 		print(labels_top2)
# 		if (pred_top2 == labels_top2).sum() == 2 or (pred_top2 == torch.index_select(labels_top2,0,torch.Tensor([1,0]).long())).sum() == 2:
# 			running_correct += 1

# test_acc = float(running_correct)/len(testLoader.dataset)
# avg_test_loss = runningLoss/float(idx + 1)
# print(avg_test_loss, test_acc * 100)

# print('Using the best')

# runningLoss = 0.0 
# avg_test_loss = 0.0
# running_correct = 0

# for idx, data in enumerate(testLoader):
# 	inputs,labels = data
# 	# print(inputs.size())
# 	# Wrap them in Variable
# 	if use_gpu:
# 		inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
# 	predicted = best_net(inputs)
# 	loss = criterion(predicted, labels)
# 	runningLoss += loss.data[0]

# 	print(predicted.size())
# 	print(labels.size())
# 	for i in range(predicted.size()[0]):
# 		pred_top2 = torch.topk(predicted[i].data.cpu(), 2)[1]
# 		labels_top2 = torch.topk(labels[i].cpu(), 2)[1]
# 		print(pred_top2)
# 		print(labels_top2)
# 		if (pred_top2 == labels_top2).sum() == 2 or (pred_top2 == torch.index_select(labels_top2,0,torch.Tensor([1,0]).long())).sum() == 2:
# 			running_correct += 1

# test_acc = float(running_correct)/len(testLoader.dataset)
# avg_test_loss = runningLoss/float(idx + 1)
# print(avg_test_loss, test_acc * 100)