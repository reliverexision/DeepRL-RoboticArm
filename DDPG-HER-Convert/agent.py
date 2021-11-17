import tensorflow as tf
import numpy as np
from models import Actor, Critic
from memory import Memory
from mpi4py import MPI
from normalizer import Normalizer

class Agent:
	def __init__(self, num_states, num_actions, num_goals, action_bounds, mem_size, env, k_future, batch_size, 
				action_size=1, tau=0.05, gamma=0.98, actor_lr=1e-3, critic_lr=1e-4):
		self.num_states    = num_states
		self.num_actions   = num_actions
		self.num_goals     = num_goals
		self.action_bounds = action_bounds
		self.mem_size      = mem_size
		self.env           = env
		self.k_future      = k_future
		self.batch_size    = batch_size
		self.action_size   = action_size
		self.tau           = tau
		self.gamma         = gamma
		self.actor_lr      = actor_lr
		self.critic_lr     = critic_lr

		# Initialise actor and critic networks
		actor_input_shape  = (num_states[0] + num_goals)
		critic_input_shape = (num_states[0] + num_goals + num_actions)
		self.actor         = Actor(actor_input_shape, num_actions, output_activation='tanh')
		self.critic        = Critic(critic_input_shape)

		self.sync_networks(self.actor.network)
		self.sync_networks(self.critic.network)

		# Initialise actor and critic target networks that are first synced to the main networks
		self.actor_target  = self.actor.init_target_network()
		self.critic_target = self.critic.init_target_network()

		# Initialise memory buffer
		self.memory        = Memory(mem_size, k_future, env)

		# Optimizer for each network
		self.actor_optimizer  = tf.keras.optimizers.Adam(learning_rate=actor_lr)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

		self.state_normalizer = Normalizer(self.num_states[0], default_clip_range=5)
		self.goal_normalizer  = Normalizer(self.num_goals, default_clip_range=5)

	def choose_action(self, state, goal, train_mode=True):
		state = self.state_normalizer.normalize(state)
		goal  = self.goal_normalizer.normalize(goal)
		state = np.expand_dims(state, axis=0)
		goal  = np.expand_dims(goal, axis=0)
		x     = np.concatenate([state, goal], axis=1)

		action = self.actor.predict(x)

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
		states, actions, rewards, next_states, goals = self.memory.sample(self.batch_size)
	
		states = self.state_normalizer.normalize(states)
		next_states = self.state_normalizer.normalize(next_states)
		goals = self.goal_normalizer.normalize(goals)
		inputs = np.concatenate([states, goals], axis=1)
		next_inputs = np.concatenate([next_states, goals], axis=1)
		inputsT = tf.convert_to_tensor(inputs)

        # tf.GradientTape() is to enable automatic differentiation
        # Actor optimization 
		with tf.GradientTape() as tape2:
			Aprime = self.actor(inputs)
			temp = tf.keras.layers.concatenate([inputsT, Aprime], axis=1)
			q_eval = self.critic(temp)
			actor_loss = -tf.reduce_mean(q_eval)
			mu_grads = tape2.gradient(actor_loss, self.actor.network.trainable_variables)
		# print("\nActor")
		self.sync_grads(mu_grads)
		self.actor_optimizer.apply_gradients(zip(mu_grads, self.actor.network.trainable_variables))

		# Critic optimization
		with tf.GradientTape() as tape:
			next_a = self.actor_target(next_inputs)
			temp = np.concatenate((next_inputs, next_a), axis=1)
			target_q = rewards + self.gamma * self.critic_target(temp)
			temp2 = np.concatenate((inputs, actions), axis=1)
			q_eval = self.critic(temp2)
			critic_loss = tf.reduce_mean((q_eval - target_q)**2)
			q_grads = tape.gradient(critic_loss, self.critic.network.trainable_variables)
		# print("\nCritic")
		self.sync_grads(q_grads)
		self.critic_optimizer.apply_gradients(zip(q_grads, self.critic.network.trainable_variables))

		# target_q = self.critic_target(next_inputs, self.actor_target(next_inputs))
		# target_returns = rewards + self.gamma * target_q
		# target_returns = tf.clip_by_value(target_returns, -1 / (1 - self.gamma), 0)

		# with tf.GradientTape() as tape2:
		# 	a = self.actor(inputs)
		# 	actor_loss = tf.reduce_mean((-self.critic(inputs,a)))
		# 	actor_loss += tf.reduce_mean(a**2)
		# 	mu_grads = tape2.gradient(actor_loss, self.actor.network.trainable_variables)
		# # self.sync_grads(self.actor.network)
		# self.actor_optimizer.apply_gradients(zip(mu_grads, self.actor.network.trainable_variables))

		# with tf.GradientTape() as tape:
		# 	q_eval = self.critic(inputs, actions)
		# 	critic_loss = tf.reduce_mean((target_returns - q_eval)**2)
		# 	q_grads = tape.gradient(critic_loss, self.critic.network.trainable_variables)
		# # self.sync_grads(self.critic.network)
		# self.critic_optimizer.apply_gradients(zip(q_grads, self.critic.network.trainable_variables))

		# Returns the individual values of actor_loss and critic_loss
		return actor_loss.numpy(), critic_loss.numpy()

	# Save actor network to a file
	def save_weights(self):
		pass
		# save_path = "FetchPickAndPlaceActor"
		# model = self.actor.network.get_weights()
		# checkpoint = tf.train.Checkpoint(models=model, 
		# 								 state_normalizer_mean=self.state_normalizer.mean,
		# 								 state_normalizer_std=self.state_normalizer.std,
		# 								 goal_normalizer_mean=self.goal_normalizer.mean,
		# 								 goal_normalizer_std=self.goal_normalizer.std)



	# Load actor network from a file
	def load_weights(self):
		# self.actor.network = tf.keras.models.load_model("FetchPickAndPlaceActor")
		pass

	def set_to_eval_mode(self):
		self.actor.network.evaluate()

	def update_networks(self):
		Actor.soft_update_network(self.actor, self.actor_target, self.tau)
		Critic.soft_update_network(self.critic, self.critic_target, self.tau)

	def _update_normalizer(self, mini_batch):
		states, goals = self.memory.sample_for_normalization(mini_batch)

		self.state_normalizer.update(states)
		self.goal_normalizer.update(goals)
		self.state_normalizer.recompute_stats()
		self.goal_normalizer.recompute_stats()

	@staticmethod
	def sync_networks(network):
		comm = MPI.COMM_WORLD
		flat_params = _get_flat_params(network)
		# weights = network.get_weights()
		comm.Bcast(flat_params, root=0)
		_set_flat_params(network, flat_params)
		# network.set_weights(weights)

	@staticmethod
	def sync_grads(gradients):
		flat_grads = _get_flat_grads(gradients)
		comm = MPI.COMM_WORLD
		global_grads = np.zeros_like(flat_grads)
		comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
		_set_flat_grads(gradients, global_grads)
		i = 0
		# for element in gradient:
		# 	print(i+1)
		# 	e_shape = element.shape
		# 	print(element.shape)
		# 	e_size = element.numpy().size
		# 	print(element.numpy().size)
		# 	i += 1
		sync_grad = gradients
		# comm = MPI.COMM_WORLD

def _get_flat_params(network):
	return np.concatenate([param.flatten() for param in network.get_weights()])

def _get_flat_grads(gradients):
	return np.concatenate([grad.numpy().flatten() for grad in gradients])

def _set_flat_params(network, flat_params):
	pointer = 0
	for i in range(len(network.layers)):
		layer = network.get_layer(index = i)
		
		# Get layer weights size and shape
		wsize = layer.kernel.numpy().size
		wshape = layer.kernel.numpy().shape
		
		# Get layer output parameters size
		osize = layer.output_shape[1]

		# Set new layer weights and outputs 
		new_weights = np.asarray(flat_params[pointer:pointer+wsize], dtype=np.float32).reshape(wshape)
		pointer += wsize
		new_out_params = np.asarray(flat_params[pointer:pointer+osize], dtype=np.float32)
		pointer += osize
		new_params = [new_weights, new_out_params]
		layer.set_weights(new_params)

def _set_flat_grads(gradients, flat_grads):
    pointer = 0
    # print("\n\n\n\nOld grad: {}".format(gradients))
    for grad in gradients:
    	# Get grad shape and size
    	gshape = grad.shape
    	gsize = grad.numpy().size

    	# Set synced grad
    	sync_grad = np.asarray(flat_grads[pointer:pointer+gsize], dtype=np.float32).reshape(gshape)
    	grad = tf.convert_to_tensor(sync_grad)
    	pointer += gsize
    # print("\nNew Grad: {}".format(gradients))