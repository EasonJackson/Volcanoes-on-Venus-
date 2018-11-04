import tensorflow as tf



'''
Input Layer --> |Convolutional   Layer 1|  --> |Convolutional    Layer 2| --> |Convolutional     Layer 3|
                |Filter      Max pooling|      |Filter       Max pooling|     |Filter        Max pooling|
110 * 110       |10 * 10     3 * 3      |      |5 * 5        3 * 3      |     |

'''

# Constants
learning_rate = 0.0001
epochs = 10
batch_size = 50

def build_input(rows, cols):
    x = tf.placeholder(tf.float32, [None, rows * cols])
    # dynamically reshape the input
    x_shaped = tf.reshape(x, [-1, rows, cols, 1])
    # now declare the output data placeholder - Yes or No
    y = tf.placeholder(tf.float32, [None, 2])

    return x_shaped, y


def build_convolutional_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    '''
    Arguments
    input_data - tensor, training set
    num_input_channels - int, number of input layer stacks
    num_filters - int, number of filters used in current layer
    filter_shape - list of int, len=2, height by width
    pool_shape - list of int, len=2, height by width
    name - string, name of current layer
    '''
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    # Strides are by default = 1
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, pool_shape[1], pool_shape[0], 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


def build_final_layer(input_data, num_input_channels, input_shape):
    w1 = tf.Variable(tf.truncated_normal([num_input_channels * input_shape[0] * input_shape[1], 1000], stddev=0.03), name='final_layer1_W')
    b1 = tf.Variable(tf.truncated_normal([1000]), name='final_layer1_b')
    final_layer1 = tf.nn.relu(tf.nn.matmul(input_data, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal([1000, 50], stddev=0.03), name='final_layer2_W')
    b2 = tf.Variable(tf.truncated_normal([50]), name='final_layer2_b')
    final_layer2 = tf.nn.relu(tf.matmul(final_layer1, W2) + b2)

    w3 = tf.Variable(tf.truncated_normal([50, 2], stddev=0.03), name='final_layer3_W')
    b3 = tf.Variable(tf.truncated_normal([2]), name='final_layer3_b')
    final_layer3 = tf.nn.softmax(tf.matmul(final_layer2, W3) + b3) 

    return final_layer3


def build_cnn():
    x_shaped, y = build_input(110, 110)
    layer1 = build_convolutional_layer(x_shaped, 1, 32, [10, 10], [3, 3], 'Layer1')
    layer2 = build_convolutional_layer(layer1, 32, 64, [5, 5], [3, 3], 'Layer2')

    output = build_final_layer(layer2, 64, [])

