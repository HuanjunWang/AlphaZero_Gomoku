import tensorflow as tf

a = tf.get_variable("a", dtype=tf.float32, shape=[1,2])
b = tf.get_variable("b", dtype=tf.float32, shape=[2,3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    vars = tf.trainable_variables()
    print(vars) #some infos about variables...
    vars_vals = sess.run(vars)
    for var, val in zip(vars, vars_vals):
        print("var: {}, value: {}".format(var.name, val)) #...or sort it in a list....


