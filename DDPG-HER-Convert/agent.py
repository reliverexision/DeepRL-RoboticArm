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
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

		self.state_normalizer = Normalizer(self.num_states[0], default_clip_range=5)
		self.goal_normalizer = Normalizer(self.num_goals, default_clip_range=5)

	def choose_action(self, state, goal, train_mode=True):
		state = self.state_normalizer.normalize(state)
		goal = self.goal_normalizer.normalize(goal)
		state = np.expand_dims(state, axis=0)
		goal = np.expand_dims(goal, axis=0)

		

		if train_mode:
			action += 0.2 * np.random.randn(self.num_actions)
			action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

			random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
											size=self.num_actions)
			action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

		return action

	def store_transition(self, mini_batch):
		for batch in mini_batch:
			self.memory.add(batch)
		self._update_normalizer(mini_batch)

	def train(self):
		pass

	# Save actor network to a file
	def save_weights(self):
		actor_filepath = "FetchPickAndPlaceActor"
		# self.actor.network.save(actor_filepath)

	# Load actor network from a file
	def load_weights(self):
		# self.actor.network = tf.keras.models.load_model("FetchPickAndPlaceActor")

	def set_to_eval_mode(self):
		pass

	def update_networks(self):
		Actor.soft_update_network(self.actor, self.actor_target, self.tau)
		Critic.soft_update_network(self.critic, self.critic_target, self.tau)

	def _update_normalizer(self, mini_batch):
		states, goals = self.memory.sample_for_normalization(mini_batch)

        self.state_normalizer.update(states)
        self.goal_normalizer.update(goals)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()