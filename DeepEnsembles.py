import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sys import platform
if platform == "darwin":
	import matplotlib  
	matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import os

constant =  0.5*np.log(2*np.pi)

class DataLoader_RegressionToy_sinusoidal():

    def __init__(self, batch_size):

        self.xs = np.expand_dims(np.linspace(-8, 8, num=1000, dtype=np.float32), -1)

        self.ys = 5*(np.sin(self.xs)) + np.random.normal(scale=1, size=self.xs.shape)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-16, 16, num=2000, dtype=np.float32), -1)

        test_ys = 5*(np.sin(test_xs)) + np.random.normal(scale=1, size=test_xs.shape)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return np.array(test_xs_standardized), np.array(test_ys)

class MLPGaussianRegressor(nn.Module):
	def __init__(self, sizes):
		'''
		The first number in sizes is the number of input nodes
		The last number in sizes is the number of output nodes 
		'''
		super(MLPGaussianRegressor, self).__init__()
		layers = []
		for i in range(len(sizes)-2):
			layers.append(nn.Linear(sizes[i], sizes[i+1]))
			layers.append(nn.ReLU())
		layers.append(nn.Linear(sizes[-2], sizes[-1]*2))
		self.net = nn.Sequential(*layers)
		self.out = sizes[-1]

	def forward(self, x):
		output = self.net(x)
		means_ = output[:, :self.out]
		vars_ = F.softplus(output[:, self.out:]) + 1e-6
		return means_, vars_

	def nll(self, means_, vars_, target):
		diff = target - means_
		nll_loss = 0.5 * torch.log(vars_) + 0.5 * torch.pow(diff, 2) / vars_ + constant
		return nll_loss.mean()

class CNNRegressor(nn.Module):
	def __init__(self):
		super(CNNRegressor, self).__init__()
		# input image is 3*64*256
		self.out = 4
		self.conv1 = nn.Conv2d(2, 6, 3, padding = 1)
		self.conv2 = nn.Conv2d(6, 16, 3, padding = 1, stride = 2)
		self.conv3 = nn.Conv2d(16, 10, 3, padding = 1, stride = 2) # 10*16*64
		self.pool = nn.MaxPool2d(2,2) 
		self.conv_bn2 = nn.BatchNorm2d(16)
		self.conv_bn3 = nn.BatchNorm2d(10)
		self.lin1 = nn.Linear(640, 128)
		self.lin_bn1 = nn.BatchNorm1d(128)
		self.lin2 = nn.Linear(128, 4*2)
		
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x))) # 6*32*128
		x = self.pool(F.relu(self.conv_bn2(self.conv2(x)))) # 16*8*32
		x = F.relu(self.conv_bn3(self.conv3(x))) # 10*4*16
		x = torch.flatten(x, start_dim = 1)
		x = F.relu(self.lin_bn1(self.lin1(x)))
		output = self.lin2(x)
		means_ = output[:, :self.out]
		vars_ = F.softplus(output[:, self.out:]) + 1e-6
		return means_, vars_

	def nll(self, means_, vars_, target):
		diff = target - means_
		#nll_loss = diff**2
		nll_loss = 0.5 * torch.log(vars_) + 0.5 * torch.pow(diff, 2) / vars_ + constant
		return nll_loss.mean()


class DeepEnsembles():
	def __init__(self, M = 5, sizes = [6, 32, 64, 128, 32, 4], regressor = "CNN"):
		self.regressor = regressor
		if self.regressor == "MLP":
			self.ensemble = [MLPGaussianRegressor(sizes) for _ in range(M)]
		else:
			self.ensemble = [CNNRegressor() for _ in range(M)]
		self.optimizers = [torch.optim.Adam(self.ensemble[i].parameters(), lr = 0.01) for i in range(M)]

	def ensemble_mean_var(self, x):
		en_mean = 0
		en_var = 0
		x = torch.FloatTensor(x)
		for model in self.ensemble:
			model.eval()
			mean, var = model(x)
			en_mean += mean
			en_var += var + mean**2
		en_mean /= len(self.ensemble)
		en_var /= len(self.ensemble)
		en_var -= en_mean**2
		return en_mean, en_var

	def train(self, data_loader, max_iter = 5000, alpha = 0.5, eps = 5e-3):
		prev_loss = 3
		average_loss = 0
		for it in range(max_iter):
			all_loss = 0
			for m in range(len(self.ensemble)):
				if(self.regressor == "CNN"): 
					_, y, img = data_loader.next_batch()
					x = torch.FloatTensor(img)
				else:
					x, y = data_loader.next_batch()
					x = torch.FloatTensor(x)
				x.requires_grad = True
				y = torch.FloatTensor(y)
				means_, vars_ = self.ensemble[m](x)
				
				loss = self.ensemble[m].nll(means_, vars_, y)
				
				# adversarial data
				loss.backward(retain_graph =True)
				x_adv = x + eps * torch.sign(x.grad)
				means_adv, vars_adv = self.ensemble[m](x_adv)
				loss_adv = self.ensemble[m].nll(means_adv, vars_adv, y)

				total_loss = alpha * loss + (1 - alpha) * loss_adv
				self.optimizers[m].zero_grad()
				total_loss.backward()

				self.optimizers[m].step()

				all_loss += total_loss.data.item()
			average_loss += all_loss/len(self.ensemble)
			if it % 2 == 0:
				print("iter: %d; loss: %2.3f"%(it, all_loss/len(self.ensemble)))
			if (it % 20 == 0 and it > 0):
				average_loss/=20
				print("iter: %d; Average loss: %2.3f"%(it, average_loss))
				if(average_loss < prev_loss):
					os.system('spd-say "Iteration ' + str(it) + ' Decreasing" ')
				else:
					os.system('spd-say "Iteration ' + str(it) + ' Increasing" ')
				prev_loss = average_loss
				average_loss = 0

	def save(self):
		if(self.regressor == "CNN"):
			if not os.path.exists("weightsCNN/"):
				os.mkdir("weightsCNN/")
			file_name = "weightsCNN/DeepEnsembles.pt"
		else:
			if not os.path.exists("weightsMLP/"):
				os.mkdir("weightsMLP/")
			file_name = "weightsMLP/DeepEnsembles.pt"
		torch.save({"model"+str(i) : self.ensemble[i].state_dict() for i in range(len(self.ensemble))}, file_name)
		print("save model to " + file_name)

	def load(self):
		try:
			if(self.regressor == "CNN"):
				file_name = "weightsCNN/DeepEnsembles.pt"
			else:
				file_name = "weightsMLP/DeepEnsembles.pt"
			checkpoint = torch.load(file_name)
			for i in range(len(self.ensemble)):
				self.ensemble[i].load_state_dict(checkpoint["model"+str(i)])
			print("load model from " + file_name)
		except:
			print("fail to load model!")

class DeepEnsemblesEstimator():
	def __init__(self, M = 5, size = [6, 32, 64, 128, 32, 4], regressor = "CNN"):
		self.regressor = regressor
		if(self.regressor == "CNN"):
			self.model = DeepEnsembles(regressor = "CNN")
		else:
			self.model = DeepEnsembles(regressor = "MLP")
		self.model.load()

	def predict(self, x):
		with torch.no_grad():
			mean, var = self.model.ensemble_mean_var(x)
		# mean rollout
		return mean, var

# test the algorithm on a toy dataset (sinusodial)
if __name__ == "__main__":
	ens = DeepEnsembles(sizes = [1, 16, 16, 1])
	loader = DataLoader_RegressionToy_sinusoidal(batch_size = 64)
	ens.load()
	ens.train(loader)
	ens.save()
	x_test, y_test = loader.get_test_data()
	plt.scatter(x_test.reshape(-1), y_test.reshape(-1))
	plt.show()
	mean, var = ens.ensemble_mean_var(x_test)
	mean = mean.detach().numpy().reshape(-1)
	var = var.detach().numpy().reshape(-1)
	x_test = x_test.reshape(-1)
	plt.plot(x_test, mean)
	plt.plot(x_test, mean + var)
	plt.plot(x_test, mean-var)
	plt.show()
