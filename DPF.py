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
									 nn.Linear(640, 64),
									 nn.ReLU())
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = 0.001)

		# observation likelihood estimator that maps states and image encodings to probabilities
		self.obs_like_estimator = nn.Sequential(nn.Linear(5+64, 64),
												nn.ReLU(),
												nn.Linear(64, 64),
												nn.ReLU(),
												nn.Linear(64, 1),
												nn.Sigmoid())
		self.obs_like_estimator_optimizer = torch.optim.Adam(self.obs_like_estimator.parameters(), lr = 0.001)

		# particle proposer that maps encodings to particles
		self.particle_proposer = nn.Sequential(nn.Linear(64, 256),
											   nn.Dropout(0.15),
											   nn.ReLU(),
											   nn.Linear(256, 128),
											   nn.ReLU(),
											   nn.Linear(128, 5))
		self.particle_proposer_optimizer = torch.optim.Adam(self.particle_proposer.parameters(), lr = 0.001)

		# motion noise generator used for motion sampling 
		self.mo_noise_generator = nn.Sequential(nn.Linear(2, 32),
												nn.ReLU(),
												nn.Linear(32, 32),
												nn.ReLU(),
												nn.Linear(32, 1))
		self.mo_noise_generator_optimizer = torch.optim.Adam(self.mo_noise_generator.parameters(), lr = 0.001)

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
		action = action[:, None, :]
		action_input = action.repeat((1, particles.shape[1], 1))
		random_input = torch.rand(action_input.shape)
		action_random = torch.cat((action_input, random_input), axis = -1)

		# estimate action noise
		delta = self.mo_noise_generator(action_random)
		delta -= delta.mean(axis = 1, keepdim=True)
		noisy_actions = action + delta
		inputs = self.transform_particles_as_input(torch.cat((particles, noisy_actions), axis = -1))
		# estimate delta and apply to current state
		state_delta = self.dynamic_model(inputs)
		if training:
			return state_delta
		else:
			return particles + state_delta.detach()

	def initial_particles(self, img):
		encoding = self.encoder(img)
		self.particles = self.propose_particles(encoding, num_particles)
		self.particle_probs = np.ones(self.particle_num) / self.particle_num

	def permute_batch(self, x, samples):
		# get shape
		batch_size = x.shape[0]
		num_particles = x.shape[1]
		sample_size = samples.shape[1]
		# compute 1D indices into the 2D array
		idx = samples + num_particles*torch.arange(batch_size)[:, None].repeat((1, sample_size))
		result = x.view(batch_size*num_particles, -1)[idx, :]
		return result

	def loop(self, particles, particle_probs, actions, imgs, training = False):
		encoding = self.encoder(imgs)
		num_proposed = int(self.particle_num * self.propose_ratio)
		num_resampled = self.particle_num - num_proposed
		batch_size = encoding.shape[0]

		if self.propose_ratio < 1.0:
			# resampling
			basic_markers = torch.linspace(0.0, (num_resampled-1.0)/num_resampled, num_resampled)
			random_offset = torch.FloatTensor(batch_size).uniform_(0.0, 1.0/num_resampled)
			markers = random_offset[:, None] + basic_markers[None, :] # shape: batch_size * num_resampled
			cum_probs = torch.cumsum(particle_probs, axis = 1)
			marker_matching = markers[:, :, None] < cum_probs[:, None, :] # shape: batch_size * num_resampled * num_particles
			samples = marker_matching.int().argmax(axis = 2).int()
			standard_particles = self.permute_batch(particles, samples)
			standard_particles_probs = torch.ones((batch_size, num_resampled))

			# motion update
			standard_particles = self.motion_update(standard_particles, actions, training)

			# measurement update
			standard_particles_probs *= self.measurement_update(encoding, standard_particles).squeeze()

		if self.propose_ratio > 0.0:
			# propose new particles
			proposed_particles = self.propose_particles(encoding, num_proposed)
			proposed_particles_probs = torch.ones((batch_size, num_proposed))

		# normalize and combine particles
		if self.propose_ratio == 1.0:
			particles = propose_particles
			particle_probs = proposed_particles_probs

		elif self.propose_ratio == 0.0:
			particles = standard_particles
			particle_probs = standard_particles_probs

		else:
			standard_particles_probs *= (1.0 * num_resampled / self.particle_num / standard_particles_probs.sum(axis = 1, keepdim=True))
			proposed_particles_probs *= (1.0 * num_proposed / self.particle_num / proposed_particles_probs.sum(axis = 1, keepdim=True))
			particles = torch.cat((standard_particles, proposed_particles), axis = 1)
			particle_probs = torch.cat((standard_particles_probs, proposed_particles_probs), axis = 1)

		# normalize probabilities
		particle_probs /= particle_probs.sum(axis = 1, keepdim = True)

		return particles, particle_probs

	def train(self, loader, max_iter=1000):
		# no motion model here...
		# train dynamic model
		mseLoss = nn.MSELoss()
		batch_size = loader.batch_size
		#TODO can train dynamic and measurement at the same time...
		print("training motion model...")
		for it in range(max_iter):
			total_loss = []
			for _, (stateAndAction, delta, _) in enumerate(loader):
				stateAndAction = torch.FloatTensor(stateAndAction)
				state = stateAndAction[..., :4]
				action = stateAndAction[..., 4:]
				state = state[:, None, :]
				state_delta = self.motion_update(state, action, training = True)
				# define loss and optimize
				self.mo_noise_generator_optimizer.zero_grad()
				self.dynamic_model_optimizer.zero_grad()
				dynamic_loss = mseLoss(state_delta.squeeze(), delta)
				dynamic_loss.backward()
				self.mo_noise_generator_optimizer.step()
				self.dynamic_model_optimizer.step()
				total_loss.append(dynamic_loss.detach().numpy())
			print("epoch: %d, loss: %2.4f" % (it, np.mean(total_loss)))

		# train measurement model
		print("training measurement model...")
		for it in range(max_iter):
			total_loss = []
			for _, (stateAndAction, delta, imgs) in enumerate(loader):
				state = stateAndAction[..., :4] + delta
				state_repeat = state[None, ...].repeat(batch_size, 1, 1)
				encoding = self.encoder(imgs)
				measurement_model_out = self.measurement_update(encoding, state_repeat).squeeze() # should be a 2d array

				stencil = torch.eye(batch_size)
				measure_loss = -torch.mul(stencil, torch.log(measurement_model_out + 1e-16)) -  \
								torch.mul(1.0-stencil, torch.log(1.0-measurement_model_out + 1e-16))/batch_size
				measure_loss_mean = measure_loss.mean()

				self.encoder_optimizer.zero_grad()
				self.obs_like_estimator_optimizer.zero_grad()
				measure_loss_mean.backward()
				self.encoder_optimizer.step()
				self.obs_like_estimator_optimizer.step()
				total_loss.append(measure_loss_mean.detach().numpy())
			print("epoch: %d, loss: %2.4f" % (it, np.mean(total_loss)))

		# train particle proposer
		print("training proposer...")
		for it in range(max_iter):
			total_loss = []
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
				total_loss.append(proposer_loss.detach().numpy())
			print("epoch: %d, loss: %2.4f" % (it, np.mean(total_loss)))

		# # end to end training
		print("end to end training...")
		for it in range(max_iter):
			total_loss = []
			for _, (stateAndAction, delta, imgs) in enumerate(loader):
				state = stateAndAction[..., :4]
				next_state = state + delta
				action = stateAndAction[..., 4:]
				state = state[:, None, :]
				particles = state.repeat((1, self.particle_num, 1))
				particle_probs = torch.ones((batch_size, self.particle_num)) / self.particle_num
				next_particles, next_particle_probs = self.loop(particles, particle_probs, action, imgs)
				
				std = 0.2
				next_state = next_state[:, None, :].repeat((1, self.particle_num, 1))
				sq_distance = (next_particles - next_state).pow(2).sum(axis = -1)
				activations = particle_probs / np.sqrt(2.0*np.pi*std**2) * torch.exp(-sq_distance / (2.0*std**2))
				e2e_loss = -torch.log(1e-16 + activations.mean(axis = -1)).mean()

				# update all parameters
				self.mo_noise_generator_optimizer.zero_grad()
				self.dynamic_model_optimizer.zero_grad()
				self.encoder_optimizer.zero_grad()
				self.obs_like_estimator_optimizer.zero_grad()
				self.particle_proposer_optimizer.zero_grad()
				e2e_loss.backward()
				self.mo_noise_generator_optimizer.step()
				self.dynamic_model_optimizer.step()
				self.encoder_optimizer.step()
				self.obs_like_estimator_optimizer.step()
				self.particle_proposer_optimizer.step()
				total_loss.append(e2e_loss.detach().numpy())
			print("epoch: %d, loss: %2.4f" % (it, np.mean(total_loss)))

	def predict(self, action, img):
		pass

	def particles_to_state(self, particles, particle_probs):
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
