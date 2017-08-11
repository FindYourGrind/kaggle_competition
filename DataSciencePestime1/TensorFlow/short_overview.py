import tensorflow as tf

a = tf.constant(2.0, shape=[], dtype=tf.float32, name="a")
x = tf.Variable(initial_value=3.0, dtype=tf.float32)
b = tf.placeholder(tf.float32, shape=[])
f = tf.add(tf.multiply(a, x), b)  # можно было написать просто f = a*x + b

with tf.Session() as session:
    tf.global_variables_initializer().run()

    result_f, result_a, result_x, result_b = session.run([f, a, x, b], feed_dict={b: -5})
    print("f = %.1f * %.1f + %.1f = %.1f" % (result_a, result_x, result_b, result_f))
    print("a = %.1f" % a.eval())

    x = x.assign_add(1.0)
    print("x = %.1f" % x.eval())