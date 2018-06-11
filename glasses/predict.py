import cv2
import numpy as np
import tensorflow as tf

# Image file of the glass
filename = 'training_data/glass4/0.jpg'

def predict():
	image_size = 128
	num_channels = 3
	images = []
	# Reading the image using OpenCV
	image = cv2.imread(filename)
	# Resizing the image to our desired size and pre-processing will be done exactly as done during training
	image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
	images.append(image)
	images = np.array(images, dtype=np.uint8)
	images = images.astype('float32')
	images = np.multiply(images, 1.0 / 255.0)
	# The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
	x_batch = images.reshape(1, image_size, image_size, num_channels)

	# Let us restore the saved model
	sess = tf.Session()
	# Step-1: Recreate the network graph. At this step only graph is created.
	saver = tf.train.import_meta_graph('glass-model.meta')
	# Step-2: Now let's load the weights saved using the restore method.
	saver.restore(sess, tf.train.latest_checkpoint('./'))

	# Accessing the default graph which we have restored
	graph = tf.get_default_graph()
	file_writer = tf.summary.FileWriter('', sess.graph)

	# Now, let's get hold of the op that we can be processed to get the output.
	# In the original network y_pred is the tensor that is the prediction of the network
	y_pred = graph.get_tensor_by_name("y_pred:0")

	# Let's feed the images to the input placeholders
	x = graph.get_tensor_by_name("x:0")
	y_true = graph.get_tensor_by_name("y_true:0")
	y_test_images = np.zeros((1, 6))

	image_shaped_input = tf.reshape(x, [1, 128, 128, 3])
	tf.summary.image('input', image_shaped_input, 10)

	# Creating the feed_dict that is required to be fed to calculate y_pred
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	result = sess.run(y_pred, feed_dict=feed_dict_testing)

	merged = tf.summary.merge_all()

	# Results
	glass_sizes = [0.25 , 0.2, 0.1, 0.25, 0.15 , 0.17 ]
	index = 0
	propability = 0
	for i in range (0 , glass_sizes.__len__()):
		if (result[0][i]>propability):
			propability = result[0][i]
			index = i

	
	return glass_sizes[index]
print(predict())	


