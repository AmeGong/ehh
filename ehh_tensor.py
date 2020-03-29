import tensorflow as tf
import numpy as np
import math

class Network():
	def __init__(self, train_batch_size, quantiles):
		self.train_batch_size = train_batch_size
		self.test_batch_size = 1
		self.quantiles = quantiles.T
		self.n_quantiles = quantiles.shape[0]

		self.hidden_config = []
		self.hidden_weights = []
		self.hidden_biases = []
		self.out_weights = None
		self.out_bias = None

		self.tf_train_samples = None
		self.tf_train_y = None
		self.tf_simulate_sample = None
		self.tf_test_simulate = None
	
	def add_source_layer(self, data_flow):
		data_flow = data_flow.T
		dimension = data_flow.shape[0]
		feed_narray = np.zeros((self.train_batch_size, dimension, self.n_quantiles, 1))
		ret = tf.placeholder(tf.float32, shape=[self.train_batch_size, dimension, self.n_quantiles, 1], name='tf_train_samples')
		for i in range(self.train_batch_size):
			feed_narray[i,:,:,1] = data_flow[:,i]-self.quantiles
		

	
	def add_hidden(self,*, input_shape, patch_size, stride, out_depth, activation = 'min', name):
		with tf.name_scope(name):
			_, input_height, n_width, input_depth = input_shape
			out_height = (input_height-patch_size)//stride + 1
			weights = tf.Variable(
				tf.truncated_normal([out_height, patch_size, n_width, input_depth, out_depth], stddev = 0.1), name = name+'_weights')
			biases = tf.Variable(tf.constant(0.1, shape=[out_height, input_depth, out_depth]), name=name+'_biases')
			self.hidden_weights.append(weights)
			self.hidden_biases.append(biases)
			self.hidden_config.append({
				'patch_size':patch_size,
				'stride':stride,
				'out_depth':out_depth,
				'activation':activation,
				'out_height':out_height,
				'input_depth':input_depth,
				'name':name
			})

	def add_output_neuron(self, data_shape, name):
		_,n_height,n_width = data_shape
		with tf.name_scope(name):
			n_inputs = n_height*n_width
			for config in self.hidden_config:
				n_inputs += config['input_depth']*config['out_depth']*config['out_height']*n_width
			self.out_weights = tf.Variable(
					tf.truncated_normal([1,n_inputs], stddev = 0.1), name = name+'_weights')
			self.out_bias = tf.Variable(
					tf.truncated_normal([1], stddev = 0.1), name = name+'_bias')


		
	def define_inputs(self, train_samples_shape, train_y_shape):
		with tf.name_scope('inputs'):
			self.tf_train_samples = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
			self.tf_train_y = tf.placeholder(tf.float32, shape=train_y_shape, name='tf_train_y')
			self.tf_simulate_samples = tf.placeholder(tf.float32, shape=(1,train_samples_shape[1]), name='tf_simulate_samples')
	
	def define_model(self):
		def calculate(data_flow, weights, biases, stride, activation):
			patch_size,_, weights_depth = weights.shape
			batchsize, input_height, input_width,input_depth = data_flow.shape
			out_height = (input_height-patch_size)//stride + 1
			result = tf.zeros([batchsize, out_height, input_width, weights_depth*input_depth])
			for b in range(batchsize):
				for i in range(input_depth):
					for j in range(weights_depth):
						for k in range(out_height):
							if(activation == 'min'):
								result[b, k, :, i*weights_depth+j] = tf.math.reduce_min(tf.multiply(data_flow[b, k*stride:k*stride+patch_size,:,i], weights[:,:,j]), axis=0)+biases[j]
							elif(activation == 'max'):
								result[b, k, :, i*weights_depth+j] = tf.math.reduce_max(tf.multiply(data_flow[b, k*stride:k*stride+patch_size,:,i], weights[:,:,j]), axis=0)+biases[j]
			
			return result

		def model(data_flow, train = True):
			data_flow = tf.reshape(data_flow,(self.train_batch_size,-1,1))
			data_flow[tf.where((data_flow - self.quantiles)<0)] = 0
			outputs_all = [data_flow]
			for i, (weights, biases, config) in enumerate(zip(self.hidden_weights, self.hidden_biases, self.hidden_config)):
				with tf.name_scope(config['name']+'_model'):
					data_flow = calculate(data_flow,weights,biases,config['stride'],config['activation'])
					outputs_all.append(data_flow)
				
			outputs_all = tf.convert_to_tensor(outputs_all)
			outputs_all = tf.reshape(outputs_all, (-1,1))
			with tf.name_scope('output_neuron'):
				result = tf.matmul(self.out_weights, outputs_all) + self.out_bias
			return result
		
		logits = model(self.tf_train_samples)
		# Training computation.
		with tf.name_scope('loss'):
			self.loss = tf.keras.losses.MeanSquaredError(logits, self.tf_train_y)
		
		# Optimizer.
		with tf.name_scope('optimizer'):
			self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

		# Predictions for the training, validation, and test data.
		with tf.name_scope('train'):
			self.train_prediction = logits

		with tf.name_scope('simulate'):
			self.test_prediction = model(self.tf_simulate_samples, train=False)
	
	def run(self, data_iterator, train_samples, train_y):
		with tf.Session(graph=tf.get_default_graph()) as session:
			tf.initialize_all_variables().run()

			### 训练
			print('Start Training')
			# batch 1000
			for i, samples, y in data_iterator(train_samples, train_y, chunkSize=self.train_batch_size):
				_, rmse = session.run(
					[self.optimizer, self.loss],
					feed_dict={self.tf_train_samples: samples, self.tf_train_y: y}
				)

				if i % 50 == 0:
					print('Minibatch loss at step %d: %f' % (i, 20*math.log10(rmse)))
			###

			# # ### 测试
			# accuracies = []
			# for i, samples, labels in data_iterator(simulate_sample, simulate_y, chunkSize=self.test_batch_size):

			# 	y_simulate = session.run(
			# 		[self.test_prediction],
			# 		feed_dict={self.tf_test_samples: samples}
			# 	)
			# 	# result = self.test_prediction.eval(feed_dict={self.tf_test_samples: samples})
			# print(' Average  Accuracy:', np.average(accuracies))
			# print('Standard Deviation:', np.std(accuracies))
		###