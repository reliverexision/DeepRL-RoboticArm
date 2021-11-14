import numpy as np
import tensorflow as tf
# import gym
# import buffer

def ANN(input_shape, layer_sizes, hidden_activation='relu', output_activation=None):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Input(shape=input_shape))
	for size in layer_sizes[: -1]:
		model.add(tf.keras.layers.Dense(units=size, activation=hidden_activation))
	model.add(tf.keras.layers.Dense(units=layer_sizes[-1], activation=output_activation))
	return model

class DDPG:
	def __init__(self, 
				 env = None,
				 num_of_episodes = 500,
				 max_steps_per_episode = 500,
				 gamma = 0.99,
				 tau = 0.001,
				 actor_lr = 1e-3,
				 critic_lr = 1e-3,
				 batch_size = 32,
				 mem_size):

		# self.env = env

		self.num_of_episodes = num_of_episodes
		self.max_steps_per_episode = max_steps_per_episode

		# Get state and action space sizes
		# self.num_states = env.observation_space.shape[0]
		# self.num_actions = env.action_space.shape[0]
		self.num_states = 15
		self.num_actions = 4

		# Get action bounds
		self.action_max = self.env.action_space.high[0]
		self.action_min = self.env.action_space.low[0]

		# Input layer shape setting
		actor_input_shape = self.num_states
		critic_input_shape = self.num_states + self.num_actions

		# Hidden layer and output layer sizes setting
		actor_layer_sizes = (1024, 512, 256, self.num_actions)		# 3 hidden, 1 output layers
		critic_layer_sizes = (512, 256, 1)							# 2 hidden, 1 output layers

		# Initialize neural networks
		self.actor = ANN(actor_input_shape, list(actor_layer_sizes), output_activation='tanh')
		self.critic = ANN(critic_input_shape, list(critic_layer_sizes))
		self.actor_target = ANN(actor_input_shape, list(actor_layer_sizes), output_activation='tanh')
		self.critic_target = ANN(critic_input_shape, list(critic_layer_sizes))

		# Copy weights from main into target networks
		self.actor_target.set_weights(self.actor.get_weights())
		self.critic_target.set_weights(self.critic.get_weights())

		# Initialize experience replay memory
		self.memory = replay_buffer(mem_size, self.num_states, self.num_actions)

		# Initialize optimizers
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

		self.noise = OUNoise(mu=np.zeros(self.num_actions))

	def get_action(self, state, noise):
		# action = self.action_max *
		pass

	def soft_update(self)

	def train(self):
		for episode in range(self.num_of_episodes):
			# Reset states
			state = self.env.reset()
			done = False

			for step in range(self.max_steps_per_episode):
				action = self.predict(state, self.noise)
				next_state, reward, done, _ = self.env.step(action)

				# Store transition in buffer
				self.memory.store_transition(state, action, reward, next_state, done)

				# Optimising actor


				# Optimising critic


				if done:
					break

				state = next_state

	def test(self):
		pass



# Init Actor-Critic and Actor-Critic targets

# Init replay buffer
class replay_buffer:
	def __init__(self, max_size, input_shape, num_actions):

	def store_transition(self, state, action, reward, next_state, done):
		pass

	def sample(self, batch_size=32):
		pass

# Define updates (soft update)

# Ornstein Uhlenbeck Process (Stochastic noise generation to use for exploration)
class OUNoise:
	def __init__(self, mu, theta=0.15, sigma=0.2, dt=1e-2, x0=None):
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.reset()

	def sample(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
			self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.mu.shape)


if __name__ == "__main__":
	a = DDPG()