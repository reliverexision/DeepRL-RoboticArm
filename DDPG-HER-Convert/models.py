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
	def __init__(self, input_shape, num_actions, layer_sizes=None, hidden_activation='relu', output_activation='tanh'):
		if layer_sizes == None:
			layer_sizes = (1024, 512, 256, num_actions)

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
		return self.network.predict(x).reshape(self.num_actions)

	def init_target_network(self):
		t_network = Actor(self.input_shape, self.num_actions, self.layer_sizes, self.hidden_activation, self.output_activation)
		t_network.network.set_weights(self.network.get_weights())

		return t_network

	@staticmethod
	def soft_update_network(actor, actor_target, tau):
		theta_mu = np.asarray(actor.network.get_weights())
		theta_muprime = np.asarray(actor_target.network.get_weights())

		actor_target.network.set_weights(tau*theta_mu + (1-tau)*theta_muprime)

class Critic:
	def __init__(self, input_shape, layer_sizes=None, hidden_activation='relu', output_activation='tanh'):
		if layer_sizes == None:
			layer_sizes = (512, 128, 1)

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
		return self.network.predict(x).reshape(1)

	def init_target_network(self):
		t_network = Critic(self.input_shape, self.layer_sizes, self.hidden_activation, self.output_activation)
		t_network.network.set_weights(self.network.get_weights())

		return t_network

	@staticmethod
	def soft_update_network(critic, critic_target, tau):
		theta_Q = np.asarray(critic.network.get_weights())
		theta_Qprime = np.asarray(critic_target.network.get_weights())

		critic_target.network.set_weights(tau*theta_Q + (1-tau)*theta_Qprime)


if __name__ == '__main__':
	X = 2
	layer_sizes = [2,2,2]

	a = ANN(X, layer_sizes, output_activation='tanh')

	# print(len(a.layers))

	# for i in range(len(a.layers)):
	# 	X = a.get_layer(index = i)
	# 	print(X.get_weights())
	# 	# param = param
	# 	print("\n\n")
	# 	print(len(X.output_shape))

	# print(np.concatenate([param.flatten() for param in a.get_weights()]))

	i = 0
	for param in a.get_weights():
		print(i+1)
		i += 1
		print(param)


	print(a.summary())