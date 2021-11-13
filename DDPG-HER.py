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
				 batch_size = 32):

		# self.env = env

		# Get state and action space sizes
		# self.num_states = env.observation_space.shape[0]
		# self.num_actions = env.action_space.shape[0]
		self.num_states = 15
		self.num_actions = 4

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
		# replay_buffer = Buffer()

		# Initialize optimizers
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

	def get_action(state, noise):
		pass

	def test

# Init Actor-Critic and Actor-Critic targets

# Init replay buffer

# Define updates (soft update)

if __name__ == "__main__":
	a = DDPG()