import sys
sys.path.append("./models")
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # van chay duoc
from classification_models import drnn_classification
import matplotlib.pyplot as plt


#gpus = tf.config.experimental.list_logical_devices('GPU')

print('check if GPU is available')
print(tf.test.is_gpu_available())


with tf.device('/gpu:0'):
  # configurations
  data_dir = "./MNIST_data" # -> change path
  n_steps = 28*28 # -> 20
  input_dims = 1  #-> 6
  n_classes = 10  # -> 1

  # model config
  cell_type = "RNN"
  assert(cell_type in ["RNN", "LSTM", "GRU"])
  hidden_structs = [20] * 9
  dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]
  assert(len(hidden_structs) == len(dilations))

  # learning config
  batch_size = 128 # nen doi lai nho hon vi data it
  learning_rate = 1.0e-3
  training_iters = batch_size * 1000#////////
  testing_step = 5000
  display_step = 100

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
  #with tf.device('/gpu:0'): #####################################################################################
  x = tf.placeholder(tf.float32, [None, n_steps, input_dims])
  y = tf.placeholder(tf.float32, [None, n_classes])
  global_step = tf.Variable(0, name='global_step', trainable=False)
  
  # build prediction graph
  print( "==> Building a dRNN with %s cells" %cell_type)
  pred = drnn_classification(x, hidden_structs, dilations, n_steps, n_classes, input_dims, cell_type)
    
  label = tf.argmax(pred,1)
  # build loss and optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
  optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost, global_step=global_step)
  
  # evaluation model
  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) # sua lai measure
  
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)

  step = 0
  train_results = []
  validation_results = []
  test_results = []


  while step * batch_size < training_iters:
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      batch_x = batch_x[:, idx_permute]
      batch_x = batch_x.reshape([batch_size, n_steps, input_dims])
 
      feed_dict = {
              x: batch_x,
            y: batch_y
      }
  
      cost_, accuracy_, step_, _ = sess.run([cost, accuracy, global_step, optimizer], feed_dict=feed_dict)
      train_results.append([step_, cost_, accuracy_])
    
      if (step + 1) % display_step == 0:
          print("Iter " + str(step + 1) + ", Minibatch Loss: " + "{:.6f}".format(cost_) \
          + ", Training Accuracy: " + "{:.6f}".format(accuracy_))
  
      if (step + 1) % testing_step == 0:
          # validation performance
          batch_x = mnist.validation.images
          batch_y = mnist.validation.labels
  
          # permute the data
          batch_x = batch_x[:, idx_permute]
          batch_x = batch_x.reshape([-1, n_steps, input_dims])
          feed_dict = {
                x: batch_x,
                y: batch_y
          }
          cost_, accuracy__, step_ = sess.run([cost, accuracy, global_step], feed_dict=feed_dict)
          validation_results.append([step_, cost_, accuracy__])
  
            # test performance
          batch_x = mnist.test.images
          batch_y = mnist.test.labels
          batch_x = batch_x[:, idx_permute]
          batch_x = batch_x.reshape([-1, n_steps, input_dims])
          feed_dict = {
               x: batch_x,
               y: batch_y
          }
          cost_, accuracy_, step_ = sess.run([cost, accuracy, global_step], feed_dict=feed_dict)
          test_results.append([step_, cost_, accuracy_])
          print("========> Validation Accuarcy: " + "{:.6f}".format(accuracy__) \
          + ", Testing Accuarcy: " + "{:.6f}".format(accuracy_))
      step += 1
    


##  ########################################################################################## PREDICT

data =  mnist.test.images
print(len(data))
data =  data[:100]
data = data[:, idx_permute]
data = data.reshape([-1, n_steps, input_dims])


gt = mnist.test.labels[:100]
feed_dict = {
        x: data
    }

pred_label = sess.run([label],feed_dict= feed_dict)
pred_label = np.resize(pred_label,(-1,1))
gt_label = np.argmax(gt,1)
loss  = np.array(train_results)
loss = loss[:,1]

plt.figure()
plt.plot(loss,'-')
plt.legend('loss values')
plt.title('loss of training')
plt.show()

plt.figure()
plt.plot(gt_label,'rx')
plt.plot(pred_label,'bo')
plt.legend(('groundtruth','predict'))
plt.title('prediction test set')
plt.show()

