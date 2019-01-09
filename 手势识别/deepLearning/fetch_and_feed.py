import tensorflow as tf

input1=tf.constant(3.0)
input2=tf.constant(5.0)
input3=tf.constant(6.0)

add=tf.add(input2,input3)
mul=tf.multiply(input1,add)

with tf.Session() as sess:
    result=sess.run([mul,add])
    print(result)



input4=tf.placeholder(tf.float32)
input5=tf.placeholder(tf.float32)
output=tf.multiply(input4,input5)
with tf.Session() as sess:
    #feed数据按字典形式传入
    print(sess.run(output,feed_dict={input4:[2.],input5:[3.]}))