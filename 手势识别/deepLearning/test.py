from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data
import model
#G:/ImageTest/GESTURE/test/
def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   plt.imshow(image)
   image = image.resize([208, 208])
   image = np.array(image)
   return image


def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   train_dir = input('请输入文件路径(结尾加上/)：')
   train, train_label = input_data.get_files(train_dir)
   # print(train)
   image_array = get_one_image(train)
   # print(image_array)
   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 10

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 208, 208, 3])
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[208, 208, 3])
       # you need to change the directories to yours.
       logs_train_dir = 'G:/Yanjiude/dog&cat_fight/logs/'

       saver = tf.train.Saver()

       print(saver)

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           print(max_index)

'''
           if max_index==0:
               print('This is a daisy with possibility %.6f' %prediction[:, 0])
           else:
               print('This is a dandelion with possibility %.6f' %prediction[:, 1])
'''
evaluate_one_image()