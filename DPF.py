import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataLoader import CartPoleDataset
from torch.utils.data import Dataset, DataLoader
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Encoder(nn.Module):
	def __init__(self, img_stack):
		super(Encoder,self).__init__()

		# self.cnn1 = nn.Sequential(nn.Conv2d(img_stack, 6, 3, padding=1),# 360*100
		# 						  nn.MaxPool2d(2, 2)) # 180*50
		# self.cnn2 = nn.Sequential(nn.Conv2d(6, 16, 3, padding = 1, stride = 2), #90*25
		# 						  nn.MaxPool2d(2, 2)) # 12*45

		self.model = nn.Sequential(nn.Conv2d(img_stack, 16, 3, padding = 1), # 360*100
								   nn.MaxPool2d(2, 2), # 180*50
								   nn.Conv2d(16, 16, 3, padding = 1, stride = 2), # 90*25
								   nn.MaxPool2d(2, 2), # 45*12
								   nn.Conv2d(16, 16, 3, padding = 1, stride = 2), # 22*6
								   nn.Conv2d(16, 8, 1),
								   nn.ReLU(),
								   nn.Conv2d(8, 1, 1),
								   nn.ReLU(),
								   nn.Flatten(),
								   nn.Linear(138, 64)
								   )
								 

		#self.attn = Self_Attn(16, 'relu')

		# self.last = nn.Sequential(nn.Conv2d(16, 10, 3, padding=1, stride=2), # 6*22
		# 						  nn.Conv2d(10, 6, 1), # 6*6*25
		# 						  nn.Conv2d(6, 1, 1),
		# 						  nn.Flatten(),
		# 						  nn.Linear(138, 64),
		# 						  #nn.ReLU()
		# 						  )

		# self.last = nn.Sequential(nn.Conv2d(16, 10, 3, padding=1, stride=2), # 6*25
		# 						  nn.Flatten(),
		# 						  nn.Linear(1500, 64),
		# 						  nn.ReLU())

	def forward(self, x):
		# import IPython
		# IPython.embed()
		# out = self.cnn1(x)
		# out = self.cnn2(out)
		# #out, _ = self.attn(out)
		# out = self.last(out)
		out = self.model(x)
		return out

def wrapAngle(x):
	return (x+np.pi) % (2*np.pi) - np.pi

learning_rate = 0.0001

class obs_like_estimator(nn.Module):
	def __init__(self):
		super(obs_like_estimator, self).__init__()
		self.like_est = nn.Sequential(nn.Linear(64+5, 32),
									  nn.ReLU(),
									  nn.Linear(32, 16),
									  nn.ReLU(),
									  nn.Linear(16, 1),
									  nn.Sigmoid())

	def forward(self, x):
		out = self.like_est(x)
		return out

class encoder(nn.Module):
	def __init__(self):
		super(encoder, self).__init__()
		self.features = nn.Sequential(nn.Conv2d(2, 16, 3, padding = 1),
									  nn.MaxPool2d(2, 2),# 180*50
									  nn.BatchNorm2d(16),
									  nn.Conv2d(16, 32, 3, padding = 1),
									  nn.MaxPool2d(2, 2),
									  nn.BatchNorm2d(32),
									  nn.Conv2d(32, 32, 3, padding = 1),
									  nn.MaxPool2d(2, 2),# 32*45*12
									  nn.BatchNorm2d(32),
									  nn.Conv2d(32, 32, 3, padding = 1),
									  nn.MaxPool2d(2, 2),# 32*6*22
									  nn.BatchNorm2d(32),
									  )
		self.last = nn.Sequential(nn.Conv2d(32, 16, 1),
								  nn.BatchNorm2d(16),
								  nn.Conv2d(16, 1, 1), # 6*22
								  nn.ReLU(),
								  nn.Flatten(),
								  nn.Linear(132, 64))

	def forward(self, x):
		feat = self.features(x)
		out = self.last(feat)
		return feat, out

class DPF():
	def __init__(self, state_dim, action_dim, observation_dim, particle_num = 16, learn_dynamic = True, image_stack = 3):
		'''
		'''
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.observation_dim = observation_dim
		self.learn_dynamic = learn_dynamic
		self.image_stack = image_stack
		self.particle_num = particle_num
		self.particles = np.zeros((self.particle_num, self.state_dim))

		self.propose_ratio = 0.0

		self.build_model()

	def build_model(self):
		# Measurement model

		# conv net for encoding the image
		self.encoder = encoder().to(device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = learning_rate)

		# observation likelihood estimator that maps states and image encodings to probabilities
		# self.obs_like_estimator = nn.Sequential(nn.Linear(5+64, 64),
		# 										nn.ReLU(),
		# 										nn.Linear(64, 64),
		# 										nn.ReLU(),
		# 										nn.Linear(64, 1),
		# 										nn.Sigmoid()).to(device)
		self.obs_like_estimator = nn.Sequential(nn.Linear(64+5, 32),
									  nn.ReLU(),
									  nn.Linear(32, 16),
									  nn.ReLU(),
									  nn.Linear(16, 1),
									  nn.Sigmoid()).to(device)
		self.obs_like_estimator_optimizer = torch.optim.Adam(self.obs_like_estimator.parameters(), lr = learning_rate)

		# particle proposer that maps encodings to particles
		self.particle_proposer = nn.Sequential(nn.Conv2d(32, 32, 3, padding = 1), # 32*6*22
									   nn.BatchNorm2d(32),
									   nn.Conv2d(32, 16, 1),
									   nn.Conv2d(16, 1, 1),
									   nn.Flatten(),
									   nn.ReLU(),
									   nn.Linear(132, 32),
									   nn.Dropout(0.05),
									   nn.ReLU(),
									   nn.Linear(32, 4)
									   ).to(device)
		self.particle_proposer_optimizer = torch.optim.Adam(self.particle_proposer.parameters(), lr = learning_rate)

		# motion noise generator used for motion sampling 
		self.mo_noise_generator = nn.Sequential(nn.Linear(2, 32),
												nn.ReLU(),
												nn.Linear(32, 32),
												nn.ReLU(),
												nn.Linear(32, 1)).to(device)
		self.mo_noise_generator_optimizer = torch.optim.Adam(self.mo_noise_generator.parameters(), lr = learning_rate)

		# transition_model maps augmented state and action to next state
		if self.learn_dynamic:
			self.dynamic_model = nn.Sequential(nn.Linear(6, 64),
												  nn.ReLU(),
												  nn.Linear(64, 128),
												  nn.ReLU(),
												  nn.Linear(128, 64),
												  nn.ReLU(),
												  nn.Linear(64, self.state_dim)).to(device)
			self.dynamic_model_optimizer = torch.optim.Adam(self.dynamic_model.parameters(), lr = learning_rate)

	def measurement_update(self, encoding, particles):
		'''
		Compute the likelihood of the encoded observation for each particle.
		'''
		particle_input = self.transform_particles_as_input(particles.to(device))
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
		batch, channel, H, W = encoding.shape
		duplicated_encoding = encoding[:, None, ...].repeat((1, num_particles, 1, 1, 1)).view((-1, channel, H, W))
		#duplicated_encoding = encoding[:, None, :].repeat((1, num_particles, 1)).to(device)
		proposed_particles = self.particle_proposer(duplicated_encoding)

		# proposed_particles = torch.cat((proposed_particles[..., 0:1],
		# 								proposed_particles[..., 1:2],
		# 								torch.atan2(proposed_particles[..., 2:3], proposed_particles[..., 3:4]),
		# 								proposed_particles[..., 4:]), axis = -1)
		return proposed_particles.view((batch, num_particles, -1))

	def motion_update(self, particles, action, training = False):
		action = action[:, None, :]
		action_input = action.repeat((1, particles.shape[1], 1))
		random_input = torch.rand(action_input.shape).to(device)
		action_random = torch.cat((action_input, random_input), axis = -1)

		# estimate action noise
		delta = self.mo_noise_generator(action_random)
		delta -= delta.mean(axis = 1, keepdim=True)
		noisy_actions = action.to(device) + delta
		inputs = self.transform_particles_as_input(torch.cat((particles.to(device), noisy_actions), axis = -1))
		# estimate delta and apply to current state
		state_delta = self.dynamic_model(inputs)
		if training:
			return state_delta
		else:
			return particles + state_delta.detach()

	def permute_batch(self, x, samples):
		# get shape
		batch_size = x.shape[0]
		num_particles = x.shape[1]
		sample_size = samples.shape[1]
		# compute 1D indices into the 2D array
		idx = samples + num_particles*torch.arange(batch_size)[:, None].repeat((1, sample_size)).to(device)
		result = x.view(batch_size*num_particles, -1)[idx, :]
		return result

	def loop(self, particles, particle_probs_, actions, imgs, training = False):
		feature, encoding = self.encoder(imgs)
		num_proposed = int(self.particle_num * self.propose_ratio)
		num_resampled = self.particle_num - num_proposed
		batch_size = encoding.shape[0]

		if self.propose_ratio == 0:
			#standard_particles = particles
			#standard_particles_probs = particle_probs_
			# motion update
			standard_particles = self.motion_update(particles, actions, training)
			if training:
				standard_particles += particles

			# measurement update
			likelihood = (self.measurement_update(encoding, standard_particles).squeeze()+1e-16)
			standard_particles_probs = likelihood * particle_probs_
		elif self.propose_ratio < 1.0:
			# resampling
			basic_markers = torch.linspace(0.0, (num_resampled-1.0)/num_resampled, num_resampled)
			random_offset = torch.FloatTensor(batch_size).uniform_(0.0, 1.0/num_resampled)
			markers = random_offset[:, None] + basic_markers[None, :] # shape: batch_size * num_resampled
			cum_probs = torch.cumsum(particle_probs_, axis = 1)
			markers = markers.to(device)
			marker_matching = markers[:, :, None] > cum_probs[:, None, :] # shape: batch_size * num_resampled * num_particles
			#samples = marker_matching.int().argmax(axis = 2).int()
			samples = marker_matching.sum(axis = 2).int()
			#print(samples)
			standard_particles = self.permute_batch(particles, samples)
			standard_particles_probs = torch.ones((batch_size, num_resampled)).to(device)

			# motion update
			if training:
				standard_particles = self.motion_update(standard_particles, actions, training) + standard_particles
			else:
				standard_particles = self.motion_update(standard_particles, actions, training)

			# measurement update
			standard_particles_probs *= (self.measurement_update(encoding, standard_particles).squeeze()+1e-16)

		if self.propose_ratio > 0.0:
			# propose new particles
			proposed_particles = self.propose_particles(feature.detach(), num_proposed)
			proposed_particles_probs = torch.ones((batch_size, num_proposed)).to(device)

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
		particle_probs /= (particle_probs.sum(axis = 1, keepdim = True))
	
		return particles, particle_probs

	def seq_train(self, loader, max_iter=1000):
		self.propose_ratio = 0.0
		for it in range(max_iter):
			sq_loss = []
			for batch, (stateAndAction, delta, imgs) in enumerate(loader):
				print(batch)
				batch_size, seq_num = imgs.shape[0], imgs.shape[1]
				state = stateAndAction[..., :4]
				#state[..., 2] = wrapAngle(state[..., 2])
				next_state = state + delta
				next_state = next_state.to(device)
				next_state[..., 2] = wrapAngle(next_state[..., 2])
				action = stateAndAction[..., 4:].to(device)

				particles = state[:, 0:1, :].to(device)
				particles = particles.repeat((1, self.particle_num, 1)).to(device)
				particle_probs = torch.ones((batch_size, self.particle_num)).to(device) / self.particle_num
				imgs = imgs.to(device)
				loss = 0

				for t in range(seq_num):
					particles, particle_probs = self.loop(particles, particle_probs, action[:, t], imgs[:, t], training = True)
					
					next_state_repeat = next_state[:, t:(t+1), :].repeat((1, self.particle_num, 1))
					sq_distance = (particles - next_state_repeat).pow(2).sum(axis = -1)
					mseloss = (particle_probs * sq_distance).sum(axis = -1).mean()
					loss += mseloss
					print("t = %d, loss = %f" % (t, mseloss.cpu().detach().numpy()))
					#particles, particle_probs = new_particles, new_particle_probs

				loss /= seq_num

				# update all parameters
				self.mo_noise_generator_optimizer.zero_grad()
				self.dynamic_model_optimizer.zero_grad()
				self.encoder_optimizer.zero_grad()
				self.obs_like_estimator_optimizer.zero_grad()
				self.particle_proposer_optimizer.zero_grad()
				loss.backward()
				self.mo_noise_generator_optimizer.step()
				self.dynamic_model_optimizer.step()
				self.encoder_optimizer.step()
				self.obs_like_estimator_optimizer.step()
				self.particle_proposer_optimizer.step()
				#total_loss.append(loss.cpu().detach().numpy())
				sq_loss.append(mseloss.cpu().detach().numpy())
				#print(mseloss.cpu().detach().numpy())
			print("epoch: %d, loss: %2.4f"  % (it, np.mean(sq_loss)))


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
				#state[..., 2] = wrapAngle(state[..., 2])
				action = stateAndAction[..., 4:]
				state = state[:, None, :]
				state, action = state.to(device), action.to(device)
				state_delta = self.motion_update(state, action, training = True)
				# define loss and optimize
				self.mo_noise_generator_optimizer.zero_grad()
				self.dynamic_model_optimizer.zero_grad()
				dynamic_loss = mseLoss(state_delta.squeeze(), delta.to(device))
				dynamic_loss.backward()
				self.mo_noise_generator_optimizer.step()
				self.dynamic_model_optimizer.step()
				total_loss.append(dynamic_loss.cpu().detach().numpy())
			print("epoch: %d, loss: %2.4f" % (it, np.mean(total_loss)))
		#self.save()
		
		# train measurement model
		print("training measurement model...")
		for it in range(max_iter):
			total_loss = []
			for _, (stateAndAction, delta, imgs) in enumerate(loader):
				batch_size = imgs.shape[0]
				state = stateAndAction[..., :4] + delta
				state[..., 2] = wrapAngle(state[..., 2])
				state_repeat = state[None, ...].repeat(batch_size, 1, 1)
				_, encoding = self.encoder(imgs.to(device))
				measurement_model_out = self.measurement_update(encoding, state_repeat).squeeze() # should be a 2d array

				temp = torch.eye(batch_size).to(device)
				measure_loss = -torch.mul(temp, torch.log(measurement_model_out + 1e-16))/batch_size - \
								torch.mul(1.0-temp, torch.log(1.0-measurement_model_out + 1e-16))/(batch_size**2-batch_size)
				measure_loss_mean = measure_loss.sum()

				self.encoder_optimizer.zero_grad()
				self.obs_like_estimator_optimizer.zero_grad()
				measure_loss_mean.backward()
				self.encoder_optimizer.step()
				self.obs_like_estimator_optimizer.step()
				total_loss.append(measure_loss_mean.cpu().detach().numpy())
			print("epoch: %d, loss: %2.4f" % (it, np.mean(total_loss)))
		#self.save()
		# train particle proposer
		print("training proposer...")
		for it in range(max_iter):
			total_loss = []
			sq_loss = []
			for _, (stateAndAction, delta, imgs) in enumerate(loader):
				state = stateAndAction[..., :4] + delta
				state[..., 2] = wrapAngle(state[..., 2])
				encoding, _ = self.encoder(imgs.to(device))
				encoding = encoding.detach()
				proposed_particles = self.propose_particles(encoding, self.particle_num)
				#proposed_particles = wrapAngle(proposed_particles)
				state = state[:, None, :].repeat((1, self.particle_num, 1))
				state = state.to(device)
				std = 0.2
				diff = proposed_particles - state
				
				#diff[..., 2] = wrapAngle(diff[..., 2])
				sq_distance = diff.pow(2).sum(axis = -1)
				activations = 1.0 / self.particle_num / np.sqrt(2.0*np.pi*std**2) * torch.exp(-sq_distance / (2.0*std**2))
				proposer_loss = -torch.log(1e-16 + activations.sum(axis = -1)).mean()
				mseloss = mseLoss(proposed_particles, state)
				self.particle_proposer_optimizer.zero_grad()
				#self.encoder_optimizer.zero_grad()
				mseloss.backward()
				self.particle_proposer_optimizer.step()
				#self.encoder_optimizer.step()
				total_loss.append(proposer_loss.cpu().detach().numpy())
				sq_loss.append(mseloss.cpu().detach().numpy())
			print("epoch: %d, loss: %2.4f, %2.4f"  % (it, np.mean(total_loss), np.mean(sq_loss)))
		#self.save()
		
		# # end to end training
		print("end to end training...")
		for it in range(max_iter):
			total_loss = []
			sq_loss = []
			for _, (stateAndAction, delta, imgs) in enumerate(loader):
				batch_size = imgs.shape[0]
				state = stateAndAction[..., :4]
				#state[..., 2] = wrapAngle(state[..., 2])
				next_state = state + delta
				next_state = next_state.to(device)
				next_state[..., 2] = wrapAngle(next_state[..., 2])
				action = stateAndAction[..., 4:]
				state = state[:, None, :]
				particles = state.repeat((1, self.particle_num, 1))
				particle_probs = torch.ones((batch_size, self.particle_num)) / self.particle_num
				particles, particle_probs, action, imgs = particles.to(device), particle_probs.to(device), action.to(device), imgs.to(device)
				next_particles, next_particle_probs = self.loop(particles, particle_probs, action, imgs, training = True)
				#next_particles_pred = self.particles_to_state(next_particles, next_particle_probs)
				std = 0.5
				
				next_state_repeat = next_state[:, None, :].repeat((1, self.particle_num, 1))
				sq_distance = (next_particles - next_state_repeat).pow(2).sum(axis = -1)
				activations = next_particle_probs / np.sqrt(2.0*np.pi*std**2) * torch.exp(-sq_distance / (2.0*std**2))
				e2e_loss = -torch.log(1e-16 + activations.sum(axis = -1)).mean()
				
				mseloss = (next_particle_probs * sq_distance).sum(axis=-1).mean()
				
				#mean_next_state = self.particles_to_state(next_particles, next_particle_probs)

				# update all parameters
				self.mo_noise_generator_optimizer.zero_grad()
				self.dynamic_model_optimizer.zero_grad()
				self.encoder_optimizer.zero_grad()
				self.obs_like_estimator_optimizer.zero_grad()
				self.particle_proposer_optimizer.zero_grad()
				mseloss.backward()
				self.mo_noise_generator_optimizer.step()
				self.dynamic_model_optimizer.step()
				self.encoder_optimizer.step()
				self.obs_like_estimator_optimizer.step()
				self.particle_proposer_optimizer.step()
				total_loss.append(e2e_loss.cpu().detach().numpy())
				sq_loss.append(mseloss.cpu().detach().numpy())
			print("epoch: %d, loss: %2.4f, %2.4f"  % (it, np.mean(total_loss), np.mean(sq_loss)))
		self.save()

	def initial_particles(self, state, img):
		state = torch.FloatTensor(state)
		img = torch.FloatTensor(img).to(device)
		self.particles = state[:, None, :].repeat((1, self.particle_num, 1)).to(device)
		self.particle_probs = torch.ones((1, self.particle_num)).to(device) / self.particle_num
		self.imgs = img[None].repeat((1, 2, 1, 1))

	def predict(self, action, img):
		img = torch.FloatTensor(img).to(device)
		# import IPython
		# IPython.embed()
		self.imgs[:, :-1, ...] = self.imgs[:, 1:, ...]
		self.imgs[:, -1, ...] = img
		action = torch.FloatTensor(action).to(device)
		with torch.no_grad():
			self.particles, self.praticles_probs = self.loop(self.particles, self.particle_probs, action, self.imgs)
		return self.particles_to_state(self.particles, self.particle_probs)

	def particles_to_state(self, particles, particle_probs):
		mean_state = particles * particle_probs[..., None]
		mean_state = mean_state.sum(axis = 1)
		return mean_state

	def load(self, file = "DPF_rec.pt"):
		try:
			if not os.path.exists("weights/"):
				os.mkdir("weights/")
			file_name = "weights/" + file
			checkpoint = torch.load(file_name)
			self.encoder.load_state_dict(checkpoint["encoder"])
			self.obs_like_estimator.load_state_dict(checkpoint["obs_like_estimator"])
			self.particle_proposer.load_state_dict(checkpoint["particle_proposer"])
			self.mo_noise_generator.load_state_dict(checkpoint["mo_noise_generator"])
			self.dynamic_model.load_state_dict(checkpoint["dynamic_model"])
			print("load model from " + file_name)
		except:
			print("fail to load model!")

	def save(self):
		if not os.path.exists("weights/"):
			os.mkdir("weights/")
		file_name = "weights/DPF.pt"
		torch.save({"encoder" : self.encoder.state_dict(),
					"obs_like_estimator" : self.obs_like_estimator.state_dict(),
					"particle_proposer" : self.particle_proposer.state_dict(),
					"mo_noise_generator" : self.mo_noise_generator.state_dict(),
					"dynamic_model" : self.dynamic_model.state_dict()}, file_name)
		print("save model to " + file_name)

if __name__ == "__main__":
	dpf = DPF(4, 1, 64, image_stack = 3)
	dpf.load()
	dataset = CartPoleDataset(need_img = True, img_stack = 2, augmented = False, seq_num = 3)
	loader = DataLoader(dataset, batch_size = 32, shuffle = True, num_workers=1)

	for i in range(1):
		print("big ephch %d" % (i))
		dpf.seq_train(loader, 20)
		dpf.save()