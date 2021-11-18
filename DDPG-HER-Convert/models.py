import tensorflow as tf
import numpy as np

def ANN(input_shape, layer_sizes, hidden_activation='relu', output_activation=None):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Input(shape=input_shape))
	for size in layer_sizes[: -1]:
		model.add(tf.keras.layers.Dense(units=size, activation=hidden_activation))
	model.add(tf.keras.layers.Dense(units=layer_sizes[-1], activation=output_activation))
	return model

class Actor:
	def __init__(self, input_shape, num_actions, layer_sizes=None, hidden_activation='relu', output_activation=None):
		if layer_sizes == None:
			layer_sizes = (256, 256, 256, num_actions)

		self.input_shape = input_shape
		self.num_actions = num_actions
		self.layer_sizes = layer_sizes
		self.hidden_activation = hidden_activation
		self.output_activation = output_activation

		self.network = ANN(input_shape, list(layer_sizes), hidden_activation, output_activation)

	# Returns tensor object
	def __call__(self, x):
		return self.network(x)

	# Returns numpy array
	def predict(self, x):
		# print(self.network.predict(x))
		return self.network.predict(x).reshape(self.num_actions)

	def init_target_network(self):
		t_network = Actor(self.input_shape, self.num_actions, self.layer_sizes, self.hidden_activation, self.output_activation)
		t_network.network.set_weights(self.network.get_weights())

		return t_network

	@staticmethod
	def soft_update_network(actor, actor_target, tau):
		theta_mu = np.asarray(actor.network.get_weights(), dtype=object)
		theta_muprime = np.asarray(actor_target.network.get_weights(), dtype=object)

		actor_target.network.set_weights(tau*theta_mu + (1-tau)*theta_muprime)

class Critic:
	def __init__(self, input_shape, layer_sizes=None, hidden_activation='relu', output_activation=None):
		if layer_sizes == None:
			layer_sizes = (256, 256, 256, 1)

		self.input_shape = input_shape
		self.layer_sizes = layer_sizes
		self.hidden_activation = hidden_activation
		self.output_activation = output_activation

		self.network = ANN(input_shape, list(layer_sizes), hidden_activation, output_activation)

	# Returns tensor object
	def __call__(self, x):
		return self.network(x)

	# Returns numpy array
	def predict(self, x, a):
		XA = np.concatenate((x, a), axis=1)
		return self.network.predict(XA).reshape(1)

	def init_target_network(self):
		t_network = Critic(self.input_shape, self.layer_sizes, self.hidden_activation, self.output_activation)
		t_network.network.set_weights(self.network.get_weights())

		return t_network

	@staticmethod
	def soft_update_network(critic, critic_target, tau):
		theta_Q = np.asarray(critic.network.get_weights(), dtype=object)
		theta_Qprime = np.asarray(critic_target.network.get_weights(), dtype=object)

		critic_target.network.set_weights(tau*theta_Q + (1-tau)*theta_Qprime)