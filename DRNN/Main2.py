import sys

sys.path.append("./models")
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # van chay duoc
from classification_models import drnn_classification
import matplotlib.pyplot as plt
import torch.nn.functional as F
# gpus = tf.config.experimental.list_logical_devices('GPU')

print('check if GPU is available')
print(tf.test.is_gpu_available())

with tf.device('/gpu:0'):
    # configurations
    data_dir = "./MNIST_data"  # -> change path
    n_steps = 20
    input_dims = 6
    n_classes = 1

    # model config
    cell_type = "RNN"
    assert (cell_type in ["RNN", "LSTM", "GRU"])
    hidden_structs = [20] * 5
    dilations = [1, 2, 4, 8, 16]
    assert (len(hidden_structs) == len(dilations))
    train = np.load('ACB.npy')
    trainlb = np.load('lbACB.npy')
    test = np.load('SLS.npy')
    testlb = np.load('lbSLS.npy')
    # learning config
    batch_size = 32
    learning_rate = 1.0e-3
    training_iters = batch_size * 40000
    testing_step = 50
    display_step = 10

    # permutation seed
    seed = 92916
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    if 'seed' in globals():
        rng_permute = np.random.RandomState(seed)
        idx_permute = rng_permute.permutation(n_steps)
    else:
        idx_permute = np.random.permutation(n_steps)
    # build computation graph
    tf.reset_default_graph()
    # with tf.device('/gpu:0'): #####################################################################################
    x = tf.placeholder(tf.float32, [None, n_steps, input_dims])
    y = tf.placeholder(tf.float32, (None,))  # [None, n_classes]) # bo n_classes
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # build prediction graph
    print("==> Building a dRNN with %s cells" % cell_type)
    pred = drnn_classification(x, hidden_structs, dilations, n_steps, n_classes, input_dims, cell_type)


    # build loss and optimizer
    cost = tf.reduce_mean(tf.abs(pred - tf.expand_dims(y,
                                                -1)))  # tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # bo cross entropy, resize y thanh (batchsize,1)
    # https://pytorch.org/docs/stable/nn.html : torch.nn.MSELoss
    # torch.abs(a)
    optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost, global_step=global_step)
    # https://cs230.stanford.edu/blog/pytorch/
    # torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    step = 0
    train_results = []
    validation_results = []
    test_results = []
    lossfunction = []
    stepextra=0
    while step * batch_size < training_iters:
        if( stepextra +batch_size > len(train)):
            stepextra = 0
        batch_x, batch_y = train[stepextra:stepextra +batch_size], trainlb[stepextra:stepextra+batch_size]
        #batch_x = batch_x[:, idx_permute]
        batch_x = batch_x.reshape([batch_size, n_steps, input_dims])

        feed_dict = {
            x: batch_x,
            y: batch_y
        }

        cost_, step_, _ = sess.run([cost, global_step, optimizer],
                                   feed_dict=feed_dict)  # bo accuracy accuracy_ ( vi tri thu 2 ), accuracy
        lossfunction.append(cost_)
        train_results.append([step_, cost_])  # , accuracy_]) # bo accuracy

        if (step + 1) % display_step == 0:
            print("Iter " + str(step + 1) + ", Minibatch Loss training : " + "{:.6f}".format(cost_) )\
                  #+ ", Training Accuracy: " + "{:.6f}".format(accuracy_))  # bo accuracy

        if (step + 1) % testing_step == 0:
            # validation performance
            batch_x = test
            batch_y = testlb

            # permute the data
            #batch_x = batch_x[:, idx_permute]
            batch_x = batch_x.reshape([-1, n_steps, input_dims])
            feed_dict = {
                x: batch_x,
                y: batch_y
            }
            cost__, step_ = sess.run([cost, global_step], feed_dict=feed_dict)  # bo acc
            validation_results.append([step_, cost_])

            # test performance
            batch_x = train
            batch_y = trainlb
            #batch_x = batch_x[:, idx_permute]
            batch_x = batch_x.reshape([-1, n_steps, input_dims])
            feed_dict = {
                x: batch_x,
                y: batch_y
            }
            cost_,step_ = sess.run([cost, global_step], feed_dict=feed_dict)  # bo acc
            test_results.append([step_, cost_])
            print("========> Validation cost: " + "{:.6f}".format(cost__) \
                  + ", Testing cost: " + "{:.6f}".format(cost_))
        step += 1
        stepextra+=1

##  ########################################################################################## PREDICT

data = test
print(len(data))
data = data
data = data.reshape([-1, n_steps, input_dims])

gt = testlb
feed_dict = {
    x: data,
    y: gt
}

prediction,loss = sess.run([pred,cost], feed_dict=feed_dict)

print('loss value: ', loss)

plt.figure()
plt.plot(lossfunction, '-')
plt.legend('loss values')
plt.title('loss of training')
plt.show()

plt.figure()
plt.plot(gt, 'r--')
plt.plot(prediction, 'b--')
plt.legend(('groundtruth', 'predict'))
plt.title('prediction test set')
plt.show()

