import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

class DPF():
	def __init__(self, state_dim, action_dim, observation_dim, particle_num = 16, learn_dynamic = False):
		'''
		'''
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.observation_dim = observation_dim
		self.learn_dynamic = learn_dynamic
		self.image_stack = 3
		self.particle_num = particle_num
		self.particles = np.zeros((self.particle_num, self.state_dim))


	def build_model(self, ):
		# Measurement model

		# conv net for encoding the image
		self.encoder = nn.Sequential(nn.Conv2d(3, 6, 3),
									 nn.MaxPool2d(2, 2),
									 nn.Conv2d(6, 16, 3),
									 nn.Flatten(), 
									 nn.Linear(16*((self.observation_dim-1)//2-1)**2, 64))
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = 0.001)

		# observation likelihood estimator that maps states and image encodings to probabilities
		self.obs_like_estimator = nn.Sequential(nn.Linear(5+64*self.image_stack, 128),
												nn.ReLU(),
												nn.Linear(128, 64),
												nn.ReLU(),
												nn.Linear(64, 1),
												nn.Sigmoid())
		self.obs_like_estimator_optimizer = torch.optim.Adam(self.obs_like_estimator.parameters(), lr = 0.001)

		# particle proposer that maps encodings to particles
		self particle_proposer = nn.Sequential(nn.Linear(128*self.image_stack, 256),
											   nn.Dropout(0.5),
											   nn.ReLU(),
											   nn.Linear(256, 128),
											   nn.ReLU(),
											   nn.Linear(128, self.state_dim))
		self.particle_proposer_optimizer = torch.optim.Adam(self.particle_proposer.parameters(), lr = 0.001)

		# we don't have a motion noise generator here 
		# transition_model maps augmented state and action to next state
		if self.learn_dynamic:
			self.transition_model = nn.Sequential(nn.Linear(6, 128),
												  nn.ReLU(),
												  nn.Linear(128, 128),
												  nn.ReLU(),
												  nn.Linear(128, self.state_dim))
			self.transition_model_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr = 0.001)

	def measurement_update(self, encoding, particles):
		'''
		Compute the likelihood of the encoded observation for each particle.
		'''
		particle_input = self.transform_particles_as_input(particles)
		encoding_input = []
		inputs = torch.cat((particle_input, encoding_input), axis = -1)

		obs_likelihood = self.obs_like_estimator(inputs)
		return obs_likelihood.detech().numpy()

	def transform_particles_as_input(self, particles):
		inputs = torch.FloatTensor(particles)
		inputs = torch.cat((inputs[:, :2],
							torch.sin(inputs[:, 2]).view((1, -1)),
							torch.cos(inputs[:, 2]).view((1, -1)),
							inputs[:, 3].view(1, -1)))
		return inputs

	def propose_particles(self, encoding, num_particles):
		duplicated_encoding = encoding.repeat((num_particles, 1))
		proposed_particles = self.particle_proposer(duplicated_encoding)
		return proposed_particles

	def motion_update(self, particles, action):
		state_input = self.transform_particles_as_input(particles)
		action_input = torch.FloatTensor(action)
		inputs = torch.cat((state_input, action_input))
		# estimate delta and apply to current state
		state_delta = self.transition_model(inputs)
		moved_particles = particles + state_delta.detech().numpy()
		return moved_particles

	def compile_training_stages(self):
		pass

	def fit(self):
		pass

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
