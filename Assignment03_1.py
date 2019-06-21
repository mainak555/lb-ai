#%% [markdown]
'''
## CNN Assignmnet with Tensorflow
Researchers at Zalando (the e-commerce company) have developed a new image classification dataset called Fashion MNIST in hopes of replacing MNIST. This new dataset contains images of various articles of clothing and accessories — such as shirts, bags, shoes, and other fashion items.

Refer link : https://github.com/zalandoresearch/fashion-mnist

The Fashion MNIST training set contains 55,000 examples, and the test set contains 10,000 examples.

Each example is a 28x28 grayscale image (just like the images in the original MNIST), associated with a label from 10 classes (t-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots).

Fashion MNIST also shares the same train-test-split structure as MNIST, for ease of use.

Your task is to design a CNN network with tensorflow to classify different colthing items correctly. Try to play around with Hyperpaparameters like no. of layers and learning rates.
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
#from tensorflow.python.framework import ops

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
#%%
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'
}
#%%
fig = plt.figure()
for i in range(12):
    plt.subplot(4, 3, i+1)
    plt.tight_layout()
    plt.imshow(x_train[i][:,:], cmap='gray', interpolation=None)
    plt.title('Label: {}'.format(label_dict[y_train[i]]))
    plt.xticks([])
    plt.yticks([])
plt.show()
#%%
#Tain Test Set

#%%
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

n_input = 784                   #data input (img shape: 28*28)
train_size = y_train.shape[0]   #(60000, )

epoc = 10
batch_size = 64
eval_frequency = 100
#%%
tf.reset_default_graph()  
tf.set_random_seed(42)
with tf.name_scope('placeholders'):
    x = tf.placeholder(tf.float32, (batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='x')
    y = tf.placeholder(tf.int64, (batch_size), name='y')
    mode = tf.placeholder(tf.string, name='mode')

with tf.name_scope('weights'):
    w_conv1 = tf.Variable( tf.truncated_normal([5, 5, NUM_CHANNELS, 32],   #5X5 filter, depth 32
        stddev=0.1, dtype=tf.float32), name='w_conv1')
    b_conv1 = tf.Variable(tf.zeros(32), dtype=tf.float32, name='b_conv1')

    w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], 
        stddev=0.1, dtype=tf.float32), name='w_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='b_conv2')

    w_fc1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE//4 * IMAGE_SIZE//4 * 64, 512],
        stddev=0.1, dtype=tf.float32), name='w_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))

    w_fc2 = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
        stddev=0.1, dtype=tf.float32), name='w_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32), name='b_fc2') 
#%%
#with tf.name_scope('network'):
def model_cnn(data, mode='test'):
    conv = tf.nn.conv2d(data, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, b_conv1))    #bias & relu added
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool.get_shape().as_list()
    print('C1.Pool Shape: {}'.format(pool_shape))

    conv = tf.nn.conv2d(pool, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, b_conv2))    
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool.get_shape().as_list()
    print('C2.Pool Shape: {}'.format(pool_shape))
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1)

    if mode == 'train':
        hidden = tf.nn.dropout(hidden, 0.5)
    return tf.matmul(hidden, w_fc2) + b_fc2

with tf.name_scope('loss'):  
    logits = model(x, 'train')  
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

with tf.name_scope('regularizer'):      #L2 regularization for the fully connected parameters
    regularize = (
        tf.nn.l2_loss(w_fc1)
        + tf.nn.l2_loss(b_fc1)
        + tf.nn.l2_loss(w_fc2)
        + tf.nn.l2_loss(b_fc2))
    loss += 5e-4 * regularize              #Adding the regularization term to the loss


with tf.name_scope('optimizer'):
    batch = tf.Variable(0, tf.float32)  #Optimizer: set up a variable that's incremented once per batch and
                                        #controls the learning rate decay.
    learning_rate = tf.train.exponential_decay(    #Decay once per epoch, using an exponential schedule starting at 0.01
        0.01,                           # Base learning rate
        batch * batch_size,             #Current index into the dataset
        train_size,                     #Decay step
        0.95,                           #Decay rate             
        staircase=True 
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/tf_logdir/mnist-fassion-train', tf.get_default_graph())
#%%
with tf.name_scope('train_prediction'):
    train_prediction = tf.nn.softmax(logits)    #Predictions for the current training minibatch

#with tf.name_scope('eval_prediction'):

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]
    )
#%%
start_time = time.time()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(int(epoc * train_size) // batch_size):
        offset = (step * batch_size) % (train_size - batch_size)
        batch_data = x_train[offset: (offset + batch_size), ...]
        batch_labels = y_train[offset: (offset + batch_size)]
        feed_dict={
            x: batch_data.reshape(batch_data.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
            y: batch_labels,
            mode: 'train'
        }
        sess.run(optimizer, feed_dict=feed_dict)

        # print("Step: %d, Loss: %f" %(i, loss))
        # train_writer.add_summary(summary, loss)

        if step % eval_frequency == 0:
            ll, lr, prediction = sess.run([loss, learning_rate, train_prediction], feed_dict=feed_dict)
            
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step: %d (epoc: %.2f), %1f ms' %
                (step, float(step) * batch_size / train_size,
                1000 * elapsed_time / eval_frequency))
            print('Minibatch Loss: %4f, Learning Rate: %6f, Error: %.1f%%' %(ll, lr, error_rate(prediction, batch_labels)))

    sess.close()
            

#%%
