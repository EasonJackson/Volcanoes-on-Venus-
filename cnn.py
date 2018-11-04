import tensorflow as tf
import pandas as pd
import numpy as py


'''
CNN structure:

|Input Layer| --> |       Convolutional Layer 1      |  --> |       Convolutional Layer 2      | --> |       Convolutional Layer 3      | --> | Flatten Layer | --> |Fully Connected 1|  --> |Fully Connected 2| --> |output|
|           |     |Filter * 16    ReLU    Max pooling|      |Filter * 32    ReLu    Max pooling|     |Filter * 64    ReLU    Max pooling|     |36 * 64 : 1000 |     |    1000 : 50    |      |     50 : 2      |
|           |     |20 * 20        --->    3 * 3      |      |10 * 10        --->    3 * 3      |     |5 * 5          --->    2 * 2      |     |      ReLU     |     |       ReLU      |      |     Softmax     |
| 110 * 110 |     |           36 * 36 * 16           |      |           12 * 12 * 32           |     |            6 * 6 * 64            |     |     1000      |     |        50       |      |        2        |
'''

# Constants
learning_rate = 0.0001
epochs = 100
batch_size = 50


def build_input(rows, cols):
    '''
    x - (, 12100)
    x_shaped - (, 110, 110, 1)
    y - (, 2)
    '''
    x = tf.placeholder(tf.float32, [None, rows * cols])
    # dynamically reshape the input
    x_shaped = tf.reshape(x, [-1, rows, cols, 1])
    # now declare the output data placeholder - Yes or No
    y = tf.placeholder(tf.float32, [None, 2])

    return x, x_shaped, y


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
    input_data_reshape = tf.reshape(input_data, [-1, num_input_channels * input_shape[0] * input_shape[1]])

    w1 = tf.Variable(tf.truncated_normal([num_input_channels * input_shape[0] * input_shape[1], 1000], stddev=0.03), name='final_layer1_W')
    b1 = tf.Variable(tf.truncated_normal([1000]), stddev=0.01, name='final_layer1_b')
    final_layer1 = tf.nn.relu(tf.nn.matmul(input_data_reshape, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal([1000, 50], stddev=0.03), name='final_layer2_W')
    b2 = tf.Variable(tf.truncated_normal([50]), stddev=0.01, name='final_layer2_b')
    final_layer2 = tf.nn.relu(tf.matmul(final_layer1, W2) + b2)

    w3 = tf.Variable(tf.truncated_normal([50, 2], stddev=0.03), name='final_layer3_W')
    b3 = tf.Variable(tf.truncated_normal([2]), stddev=0.01, name='final_layer3_b')
    final_layer3 = tf.matmul(final_layer2, W3) + b3
    y_predict = tf.softmax(final_layer3)

    return final_layer1, final_layer2, final_layer3, y_predict


def build_cnn():
    # Input layers
    x, x_shaped, y = build_input(110, 110)

    # Convolutional layers
    conv_layer1 = build_convolutional_layer(x_shaped, 1, 16, [20, 20], [3, 3], 'Conv_Layer1')
    conv_layer2 = build_convolutional_layer(conv_layer1, 16, 32, [10, 10], [3, 3], 'Conv_Layer2')
    conv_layer3 = build_convolutional_layer(conv_layer2, 32, 64, [5, 5], [2, 2], 'Conv_Layer3')

    # Fully connected layers
    final_layer1, final_layer2, final_layer3, y_predict = build_final_layer(conv_layer3, 64, [6, 6])

    # Loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_layer3, labels=y))
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def read_input_label(image_file, label_file):
    image = pd.read_csv(image_file, header=None)
    label = pd.read_csv(label_file, header=None).loc[:,0]

    return image, label


def next_batch(df_image, df_label, batch_size=batch_size):
    batch_index = np.random.choice(len(df_image), batch_size, replace=False)

    batch_x = df_image.loc[batch_index].values()
    batch_y = df_label.loc[batch_index].values()

    return batch_x, batch_y    



def run():
    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    training_image, training_label = read_input_label('./volcanoesvenus/Volcanoes_train/train_images.csv', './volcanoesvenus/Volcanoes_train/train_labels.csv')
    texting_image, testing_label = read_input_label('./volcanoesvenus/Volcanoes_train/test_images.csv', './volcanoesvenus/Volcanoes_train/test_labels.csv')

    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(training_label) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = next_batch(training_image, training_label, batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            
            if epoch % 10 == 0:
                print('Epoch trained: {0}. Current average cost: {1}\n'.format(epoch, avg_cost))

        print("\nTraining complete!")
        


def test():

    test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})



