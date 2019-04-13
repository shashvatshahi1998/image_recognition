'''Trains and evaluates the fully connected neural net for CIFAR-10'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

'''necessary imports'''

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os.path
import two_layer_fc

# Model parameters as external flags (used for parsing alternative for arg-parse)
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate',0.001.'Learning rate for training')
flags.DEFINE_integer('max_steps',2000,'Number of steps to run trainer')
flags.DEFINE_integer('hidden1',120,'Number of units in hidden layer 1')
flags.DEFINE_integer('batch_size',400,'Batch_size. Must divide dataset sizes without remaindeer')

# Batch size should divide data without leaving any remainder

flags.DEFINE_string('train_dir','tf_logs','Directory to put the training data')
flags.DEFINE_float('reg_constant',0.1,'Regularization constant')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr, value))
print()

IMAGE_PIXELS = 3072
CLASSES = 10

begin_time = time.time()

# Put logs for each run in seperate directory


log_dir = Flags.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

# use strftime function to set the format of expressing date-time according to yourself
#Load CIFAR-10 data

data_sets = data_helpers.load_data()

# prepare tensorflow graph
#input placeholders

images_placeholder = tf.placeholder(tf.float32,shape=[None,IMAGE_PIXELS])
labels_placeholder = tf.placeholder(tf.int64,shape=[None],name='image-labels')

# Operation for classifier's result
logits = two_layer_fc.inference(image_placeholder,IMAGE_PIXEL,Flags.hidden1,CLASSES,reg_constant=Flags.reg_constant)
# Operation for calculating loss
loss = two_layer_fc.loss(logits,labels_placeholder)
# Operation for training_step
train_step = two_layer_fc.training(loss,Flags.learning_rate)
# Operation for calculating accuracy of our predictions 
accuracy = two_layer_fc.evaluation(logits,labels_placeholder)
# used for merging all the summaries at one place
summary = tf.summary.merge_all()
saver = tf.train.Saver()

# use tf.session() to run tensorflow graph.

with tf.Session() as sess:
    sess.run(tf.global_variable_initializer())
    summary_writer = tf.summary.FileWriter(logdir,sess.graph)

    zipped_data = zip(data_sets['images_train'],data_sets['label_train'])
    batches = data_helpers.gen_batch(list(zipped_data),FLAGS.batch_size,FLAGS.max_steps)

    for i range(FLAGS.max_steps):
        batch = next(batches)
        images_batch,labels_batch = zip(*batch)
        feed_dict = {images_placeholder : images_batch,
                     labels_placeholder : labels_batch}
        
        
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy,fees_dict=feed_dict)
            print('Step {:d},training_accuracy{:g}'.format(i,training_accuracy))
            summary_str = sess.run(summary,feed_dict=feed_dict)
            summary_writer.add_summary(summary_str,i)
# Perform a single training step
        sess.run([train_step,loss],feed_dict=feed_dict)
# Periodically save checkpoint

        if (i+1) % 1000 == 0:
            checkpoint_file = os.path.join(FLAGS.train_dir,'checkpoint')
            saveer.save(sess,checkpoint_file,global_step=i)
            print('Saved Checkpoint')
# Afer finishing training step , evaluate test set
        test_accuracy = sess.run(accuracy,feed_dict={images_placeholder:data_sets['images_test'],labels_placeholder:data_sets['labels_test']})
        print('Test accuracy {:g}'.format(test_accuracy))

end_time = time.time()
print('Total time: {:5.2f}s'.format(end_time - begin_time))
                  
