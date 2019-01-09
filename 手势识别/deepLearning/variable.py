import tensorflow as tf


x=tf.Variable([1,2])
a=tf.constant([3,3])

sub=tf.subtract(x,a)

add=tf.add(x,a)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(add))
    print(sess.run(sub))
#创建一个变量初始化为0
state=tf.Variable(0 ,name='counter')
new_state=tf.add(state,1)
#赋值的方法
update=tf.assign(state,new_state)

#变量需要初始化
inita=tf.global_variables_initializer()

with tf.Session() as se:
    se.run(inita)
    print(se.run(state))
    for _ in range(5):
        se.run(update)
        print(se.run(state))