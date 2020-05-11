import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataLoader import CartPoleDataset
from torch.utils.data import Dataset, DataLoader

class DPF():
	def __init__(self, state_dim, action_dim, observation_dim, particle_num = 16, learn_dynamic = True):
		'''
		'''
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.observation_dim = observation_dim
		self.learn_dynamic = learn_dynamic
		self.image_stack = 3
		self.particle_num = particle_num
		self.particles = np.zeros((self.particle_num, self.state_dim))

		self.propose_ratio = 0.7

		self.build_model()

	def build_model(self):
		# Measurement model

		# conv net for encoding the image
		self.encoder = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), # 256*64
									 nn.MaxPool2d(2, 2), # 128*32
									 nn.Conv2d(6, 16, 3, padding=1, stride = 2), #64*16
									 nn.MaxPool2d(2, 2), # 16*32*8
									 nn.Conv2d(16, 10, 3, padding=1, stride = 2), # 10*16*4
									 nn.Flatten(), 
									 nn.Linear(640, 64))
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = 0.001)

		# observation likelihood estimator that maps states and image encodings to probabilities
		self.obs_like_estimator = nn.Sequential(nn.Linear(5+64, 128),
												nn.ReLU(),
												nn.Linear(128, 64),
												nn.ReLU(),
												nn.Linear(64, 1),
												nn.Sigmoid())
		self.obs_like_estimator_optimizer = torch.optim.Adam(self.obs_like_estimator.parameters(), lr = 0.001)

		# particle proposer that maps encodings to particles
		self.particle_proposer = nn.Sequential(nn.Linear(64, 256),
											   nn.Dropout(0.5),
											   nn.ReLU(),
											   nn.Linear(256, 128),
											   nn.ReLU(),
											   nn.Linear(128, 5))
		self.particle_proposer_optimizer = torch.optim.Adam(self.particle_proposer.parameters(), lr = 0.001)

		# we don't have a motion noise generator here 
		# transition_model maps augmented state and action to next state
		if self.learn_dynamic:
			self.dynamic_model = nn.Sequential(nn.Linear(6, 64),
												  nn.ReLU(),
												  nn.Linear(64, 128),
												  nn.ReLU(),
												  nn.Linear(128, 64),
												  nn.ReLU(),
												  nn.Linear(64, self.state_dim))
			self.dynamic_model_optimizer = torch.optim.Adam(self.dynamic_model.parameters(), lr = 0.001)

	def measurement_update(self, encoding, particles):
		'''
		Compute the likelihood of the encoded observation for each particle.
		'''
		particle_input = self.transform_particles_as_input(particles)
		encoding_input = encoding[:, None, :].repeat((1, particle_input.shape[1], 1))
		inputs = torch.cat((particle_input, encoding_input), axis = -1)
		obs_likelihood = self.obs_like_estimator(inputs)
		return obs_likelihood

	def transform_particles_as_input(self, particles):
		inputs = torch.cat((particles[..., :2],
							torch.sin(particles[..., 2])[..., None],
							torch.cos(particles[..., 2])[..., None],
							particles[..., 3:]), axis = -1)
		return inputs

	def propose_particles(self, encoding, num_particles):
		duplicated_encoding = encoding[:, None, :].repeat((1, num_particles, 1))
		proposed_particles = self.particle_proposer(duplicated_encoding)
		proposed_particles = torch.cat((proposed_particles[..., :2],
										torch.atan2(proposed_particles[..., 2:3], proposed_particles[..., 3:4]),
										proposed_particles[..., 4:]), axis = -1)
		return proposed_particles

	def motion_update(self, particles, action, training = False):
		inputs = self.transform_particles_as_input(torch.cat((particles, action), axis = -1))
		# estimate delta and apply to current state
		state_delta = self.dynamic_model(inputs)
		if training:
			return state_delta
		else:
			return particles + state_delta.detach()

	def loop(self, particles, particle_probs, ):
		num_proposed = int(self.particle_num * self.propose_ratio)
		num_resampled = self.particle_num - num_proposed

		if self.propose_ratio < 1.0:
			# resampling

		if self.propose_ratio > 0.0:
			# propose new particles
			proposed_particles = self.propose_particles()

	def train(self, loader, max_iter=1000):
		# no motion model here...
		# train dynamic model
		mseLoss = nn.MSELoss()
		batch_size = loader.batch_size
		#TODO can train dynamic and measurement at the same time...

		# for it in range(max_iter):
		# 	for _, (stateAndAction, delta, _) in enumerate(loader):
		# 		stateAndAction = torch.FloatTensor(stateAndAction)
		# 		state = stateAndAction[..., :4]
		# 		action = stateAndAction[..., 4:]
		# 		state_delta = self.motion_update(state, action, training = True)
		# 		# define loss and optimize
		# 		self.dynamic_model_optimizer.zero_grad()
		# 		dynamic_loss = mseLoss(state_delta, delta)
		# 		dynamic_loss.backward()
		# 		self.dynamic_model_optimizer.step()
		# 	print(dynamic_loss)

		# train measurement model
		# for it in range(max_iter):
		# 	for _, (stateAndAction, delta, imgs) in enumerate(loader):
		# 		state = stateAndAction[..., :4] + delta
		# 		state_repeat = state[None, ...].repeat(batch_size, 1, 1)
		# 		encoding = self.encoder(imgs)
		# 		measurement_model_out = self.measurement_update(encoding, state_repeat).squeeze() # should be a 2d array

		# 		stencil = torch.eye(batch_size)
		# 		measure_loss = -torch.mul(stencil, torch.log(measurement_model_out)) -  \
		# 						torch.mul(1-stencil, torch.log(1-measurement_model_out)) / batch_size
		# 		measure_loss = measure_loss.mean()
		# 		self.obs_like_estimator_optimizer.zero_grad()
		# 		measure_loss.backward()
		# 		self.obs_like_estimator_optimizer.step()
		# 	print(measure_loss)

		# train particle proposer
		for it in range(max_iter):
			for _, (stateAndAction, delta, imgs) in enumerate(loader):
				state = stateAndAction[..., :4]
				encoding = self.encoder(imgs).detach()
				proposed_particles = self.propose_particles(encoding, self.particle_num)
				state = state[:, None, :].repeat((1, self.particle_num, 1))
				std = 0.2
				sq_distance = (proposed_particles - state).pow(2).sum(axis = -1)
				activations = 1.0 / np.sqrt(2.0*np.pi*std**2) * torch.exp(-sq_distance / (2.0*std**2))
				proposer_loss = -torch.log(1e-16 + activations.mean(axis = -1)).mean()
				self.particle_proposer_optimizer.zero_grad()
				proposer_loss.backward()
				self.particle_proposer_optimizer.step()
			print(proposer_loss)

		# end to end training

		
	def predict(self):
		pass

	def loop(self):
		pass

	def particles_to_state(self):
		pass

	def load(self):
		pass

	def save(self):
		pass

if __name__ == "__main__":
	dpf = DPF(4, 1, 64)
	dataset = CartPoleDataset(need_img = True, augmented = False)
	loader = DataLoader(dataset, batch_size = 32, shuffle = True, num_workers=4)
	it = iter(loader)
	(stateAndAction, delta, imgs) = next(it)
	xstate = stateAndAction[..., :4]
	action = stateAndAction[..., -1]
	import IPython
	IPython.embed()
