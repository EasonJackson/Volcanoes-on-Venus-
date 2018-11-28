import tensorflow as tf
import pandas as pd
import numpy as np
import sys


'''
CNN structure:

|Input Layer| --> |       Convolutional Layer 1     |  --> |       Convolutional Layer 2      | --> | Flatten Layer |  --> |Fully Connected 1| --> |output|
|           |     |Filter * 4    ReLU    Max pooling|      |Filter *  4    ReLu    Max pooling|     | 169 * 4 : 50  |      |     50 : 2      |     |      |
|           |     |10 * 10        --->   3 * 3      |      | 5 *  5        --->    3 * 3      |     |      ReLU     |      |     Softmax     |     |      |
| 110 * 110 |     |           37 * 37 *  4          |      |           13 * 13 *  4           |     |      50       |      |        2        |     |      |
'''

# Constants
learning_rate = 0.0001
epochs = 30
batch_size = 20


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

    Return
    out_layer - output layer tensor
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
    '''
    Arguments
    input_data - tensor, input layer
    num_input_channels - int, number of input layer channels
    input_shape - list of int, len=2, shape of input layer

    Return
    final_layer1 - the first dense layer
    final_layer2 - the second dense layer, also the final properbility before normalization
    y_predict - the softmax probability layer
    '''
    input_data_reshape = tf.reshape(input_data, [-1, num_input_channels * input_shape[0] * input_shape[1]])

    w1 = tf.Variable(tf.truncated_normal([num_input_channels * input_shape[0] * input_shape[1], 1000], stddev=0.03), name='final_layer1_W')
    b1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='final_layer1_b')
    final_layer1 = tf.nn.relu(tf.matmul(input_data_reshape, w1) + b1)
    print('final_layer1' + str(final_layer1.shape))

    w2 = tf.Variable(tf.truncated_normal([1000, 50], stddev=0.03), name='final_layer2_W')
    b2 = tf.Variable(tf.truncated_normal([50], stddev=0.01), name='final_layer2_b')
    final_layer2 = tf.nn.relu(tf.matmul(final_layer1, w2) + b2)
    print('final_layer2' + str(final_layer2.shape))

    w3 = tf.Variable(tf.truncated_normal([50, 2], stddev=0.03), name='final_layer3_W')
    b3 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name='final_layer3_b')
    final_layer3 = tf.nn.relu(tf.matmul(final_layer2, w3) + b3)
    print('finale_layer3' + str(final_layer3.shape))    

    y_predict = tf.nn.softmax(final_layer3)

    return final_layer1, final_layer2, final_layer3, y_predict


def build_cnn():
    '''
    Return
    x - input images
    y - input labels
    y_predict - predict labels
    cross_entropy - cross entropy as loss function
    optimiser - optimisation funtion, Adam optimizer
    confusion_mat - confusion matrix as evaluation of model
    '''
    # Input layers
    x, x_shaped, y = build_input(110, 110)
    print("X: " + str(x.shape))
    print("X_shaped: " + str(x_shaped.shape))
    print("Y: " + str(y.shape))

    # Convolutional layers
    conv_layer1 = build_convolutional_layer(x_shaped, 1, 8, [5, 5], [2, 2], 'Conv_Layer1')
    conv_layer2 = build_convolutional_layer(conv_layer1, 8, 16, [5, 5], [2, 2], 'Conv_Layer2')
    print("Conv Layer1: " + str(conv_layer1.shape))
    print("Conv Layer2: " + str(conv_layer2.shape))

    # Fully connected layers
    final_layer1, final_layer2, final_layer3, y_predict = build_final_layer(conv_layer2, 16, [28, 28])

    # Loss and optimizer
    class_weights = tf.constant([1.0, 5.0], dtype=tf.float32)
    weights = tf.reduce_sum(class_weights * y, axis=1)
    unweight_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer3, labels=y)
    cross_entropy = tf.reduce_mean(tf.math.multiply(unweight_loss, weights))
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # define an accuracy assessment operation
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion_mat = tf.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(y_predict, 1), name='Confusion_Matrix')

    return x, y, y_predict, cross_entropy, optimiser, confusion_mat


def read_input_label(image_file, label_file):
    '''
    Arguments
    image_file - string, image filename
    label_file - string, label filename
    '''
    image = pd.read_csv(image_file, header=None)
    label = pd.read_csv(label_file)[['Volcano?']]

    return image, label


def next_batch_all_positive(df_image, df_label, batch_size=batch_size, whole=False):
    '''
    Generate random mini batch with all positive data

    Arguments
    df_image - dataframe, images
    df_label - dataframe, labels
    batch_size - int, size of mini batch, default=30
    whole - bool, flag indicating select whole set as mini batch

    Return
    batch_x - sampled images
    batch_y - sampled labels
    '''
    select = df_label['Volcano?'] == 1
    df_image_true = df_image[select]
    df_label_true = df_label[select]
    
    batch_index = 0
    if not whole:
        batch_index = np.random.choice(df_label_true.index, batch_size, replace=False)
    else:
        batch_index = df.label_true.index

    batch_x = df_image.loc[batch_index].values / 255.0
    flag_y = df_label.loc[batch_index].values
    batch_y = np.zeros((batch_size, 2))
    batch_y[:, 1] = 1 # Col #0 - no volcano; Col #1 - with volcano

    return batch_x, batch_y    

def next_batch(df_image, df_label, batch_size=batch_size, whole=False):
    '''
    Generate random mini batch

    Arguments
    df_image - dataframe, images
    df_label - dataframe, labels
    batch_size - int, size of mini batch, default=30
    whole - bool, flag indicating select whole set as mini batch

    Return
    batch_x - sampled images
    batch_y - sampled labels
    '''
    true_index = df_label['Volcano?'] == 1
    df_image_true = df_image[true_index]
    df_label_true = df_label[true_index]

    df_image_false = df_image[~true_index]
    df_label_false = df_image[~true_index]

    batch_index = 0
    if not whole:
        batch_index_true = np.random.choice(df_label_true.index, batch_size // 2, replace=False)
        batch_index_false = np.random.choice(df_label_false.index, batch_size // 2, replace=False)
        batch_index = np.concatenate((batch_index_true, batch_index_false), axis=None)
    else:
        batch_index = np.arange(len(df_image))

    batch_x = df_image.loc[batch_index].values / 255.0
    flag_y = df_label.loc[batch_index].values
    batch_y = np.zeros((batch_size, 2))
    batch_y[range(batch_size), flag_y.flatten()] = 1 # Col #0 - no volcano; Col #1 - with volcano

    return batch_x, batch_y    



def run():
    '''
    Entry of running model
    '''
    # Load training and testing data sets
    try:
        training_image, training_label = read_input_label('./volcanoesvenus/Volcanoes_train/train_images.csv', './volcanoesvenus/Volcanoes_train/train_labels.csv')
        testing_image, testing_label = read_input_label('./volcanoesvenus/Volcanoes_test/test_images.csv', './volcanoesvenus/Volcanoes_test/test_labels.csv')
    except Exception:
        print("Reading training and testing datasets failed. Exit.")
        return

    print('Finishing reading datasets.')

    # Construct model
    x, y, y_predict, cross_entropy, optimiser, confusion_mat = build_cnn()
    print('Finishing build cnn.')

    # Setup the initialisation operator
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        train(sess, x, y, training_image, training_label, cross_entropy, optimiser, y_predict, confusion_mat)
        test(sess, x, y, testing_image, testing_label, confusion_mat, y_predict)
        


def train(sess, x, y, training_image, training_label, cross_entropy, optimiser, y_predict, confusion_mat):
    '''
    Training function

    Arguments
    sess - Tensorflow session
    x - input layer
    y - lable layer
    training_image - dataframe, training image set
    training_label - dataframe, training labels
    cross_entropy - loss function layer
    optimiser - optimise function layer
    '''
    total_batch = int(len(training_label) / batch_size) // 2
    #total_batch = 5
    for epoch in range(1, epochs + 1):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = next_batch(training_image, training_label, batch_size=batch_size)
            #print(batch_x)
            #print(batch_y)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        
        if epoch % 10 == 0:
            print('Epoch trained: {0}. Current average cost: {1}\n'.format(epoch, avg_cost))
            batch_x, batch_y = next_batch(training_image, training_label, batch_size=100)
            confusion_tmp = sess.run(confusion_mat, feed_dict={x:batch_x, y:batch_y})
            print(confusion_tmp)

    print("Training complete!\n")


def test(sess, x, y, testing_image, testing_label, confusion_mat, y_predict):
    '''
    Test function

    Arguments
    sess - Tensorflow session
    x - input layer
    y - lable layer
    testing_image - dataframe, testing image set
    testing_label - dataframe, testing labels
    confusion_mat - confusion matrix
    y_predict - predict labels
    '''
    test_batch_x, test_batch_y = next_batch(testing_image, testing_label, batch_size=len(testing_image), whole=True)
    test_acc = sess.run(confusion_mat, feed_dict={x: test_batch_x, y: test_batch_y})
    #y_ = y_predict.eval()
    print("Test confusion mat:")
    print(test_acc)


if __name__ == "__main__":
#    debug = False
#    if len(sys.argv) > 1:
#        if sys.argv[1] == '-d':
#            debug = True
#
#    if debug:
#        epochs = 1

    run()




