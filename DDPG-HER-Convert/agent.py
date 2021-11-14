import tensorflow as tf
import numpy as np
from models import Actor, Critic
from replay_buffer import buffer
from mpi4py import MPI
from normalizer import Normalizer

class Agent:
	def __init__(self, num_states, num_actions, num_goals, action_bounds, mem_size, env, k_future, batch_size, 
				action_size=1, tau=0.05, gamma=0.99, actor_lr=1e-3, critic_lr=1e-4):
		self.num_states = num_states
		self.num_actions = num_actions
		self.num_goals = num_goals
		self.action_bounds = action_bounds
		self.mem_size = mem_size
		self.env = env
		self.k_future = k_future
		self.batch_size = batch_size
		self.action_size = action_size
		self.tau = tau
		self.gamma = gamma
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr

		# Initialise actor and critic networks
		actor_input_shape = (num_states + num_goals)
		critic_input_shape = (num_states + num_goals + num_actions) 
		self.actor = Actor(actor_input_shape, num_actions)
		self.critic = Critic(critic_input_shape)

		# Initialise actor and critic target networks that are first synced to the main networks
		self.actor_target = self.actor.init_target_network()
		self.critic_target = self.critic.init_target_network()

		# Initialise memory buffer
		self.memory = Memory(capacity, k_future, env)

		# Optimizer for each network
		actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
		critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

	def choose_action(self):
		pass

	def store_transition(self):
		pass

	@staticmethod
	def hard_update_networks(local_model, target_model):
		pass

	@staticmethod
	def soft_update_networks(local_model, target_model, tau=0.05):
		pass

	def train(self):
		pass

	def save_weights(self):
		pass

	def load_weights(self):
		pass

	def set_to_eval_mode(self):
		pass

	def update_networks(self):
		pass

	def _update_normalizer(self, mini_batch):
		pass