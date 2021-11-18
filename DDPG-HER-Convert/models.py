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


if __name__ == '__main__':
	X = 4
	layer_sizes = [4,2,1]

	a = ANN(X, layer_sizes, output_activation='tanh')

	# states = [4, 2]
	# goals = [3]
	# states = np.expand_dims(states, axis=0)
	# goals  = np.expand_dims(goals, axis=0)
	# Aprime = [2]
	# Aprime = np.expand_dims(Aprime, axis=0)
	# Aprime = tf.convert_to_tensor(Aprime)

	# inputs = np.concatenate([states, goals], axis=1)
	# print(inputs)
	# inputsT = tf.convert_to_tensor(inputs)
	# print(inputsT)

	# temp = tf.keras.layers.concatenate([inputsT, Aprime], axis=1)
	# print(temp)

	# print(a(temp))

	# flat_param = np.random.randn(18)
	# print(flat_param)
	inputs = [4, 2, 1, 4]
	inputs = np.expand_dims(inputs, axis=0)

	# with tf.GradientTape() as tape:
	# 		critic_loss = 0.5
	# 		critic_loss = tf.convert_to_tensor(critic_loss)
	# 		q_grads = tape.gradient(critic_loss, a.trainable_variables)
	# 		print(type(q_grads))

	# for i in range(len(a.layers)):
	# 	layer = a.get_layer(index = i)
	# 	print(i+1)
	# 	param = layer.get_weights()
	# 	print(layer.output_shape)
	# 	print(param)
	# 	print("\n")
	# 	layer.set_weights(newparams)

	print(a.summary())
	print("\n")

	for i in range(len(a.layers)):
		layer = a.get_layer(index = i)
		print(i+1)
		print(layer.output_shape)
		print(layer.kernel.numpy().size)
		lsize = layer.output_shape[1]
		print(layer.get_weights())
		print("\n")
		# print(lsize)
		# a = np.asarray(layer.get_weights())
		# print(a.shape)
		# print(a)
		# print(layer.output_shape)
		# print(type(new_params))
		# print(new_params)
		# layer.set_weights(new_params)
		# pointer += lsize