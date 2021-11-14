import tensorflow as tf

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

	def init_target_network(self):
		t_network = Actor(self.input_shape, self.num_actions, self.layer_sizes, self.hidden_activation, self.output_activation)
		t_network.network.set_weights(self.network.get_weights())

		return t_network

	@staticmethod
	def soft_update_network(actor, actor_target, tau):
		theta_mu = actor.network.get_weights()
		theta_muprime = actor_target.network.get_weights()

		actor_target.network.set_weights(tau*theta_mu + (1-tau)*theta_muprime)

class Critic:
	def __init__(self, input_shape, layer_sizes=None, hidden_activation='relu', output_activation='tanh'):
		if layer_sizes == None:
			layer_sizes = (512, 128, 1)

		self.input_shape = input_shape
		self.num_actions = num_actions
		self.hidden_activation = hidden_activation
		self.output_activation = output_activation

		self.network = ANN(input_shape, list(layer_sizes), hidden_activation, output_activation)

	def init_target_network(self):
		t_network = Actor(self.input_shape, self.layer_sizes, self.hidden_activation, self.output_activation)
		t_network.network.set_weights(self.network.get_weights())

		return t_network

	@staticmethod
	def soft_update_network(critic, critic_target, tau):
		theta_Q = critic.network.get_weights()
		theta_Qprime = critic_target.network.get_weights()

		critic_target.network.set_weights(tau*theta_Q + (1-tau)*theta_Qprime)


if __name__ == '__main__':
	a = Actor()
	t_a = a.init_target_network()

	print(a.network.get_weights())
	print(t_a.network.get_weights())