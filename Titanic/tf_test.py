import titanic_preprocessor as preproc
import tensorflow as tf

input_x, input_y, output_x, output_y = preproc.get_train_data(0.7)


num_features = input_x.shape[1]
num_classes = output_x.shape[1]

X = tf.placeholder("float", [None, num_features])
Y = tf.placeholder("float", [None, num_classes])

W = tf.Variable(tf.zeros([num_features,num_classes]))
B = tf.Variable(tf.zeros([num_classes]))

pY = tf.nn.softmax(tf.matmul(X, W) + B)

#cost_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pY, logits=Y))

cost_fn = -tf.reduce_sum(Y * tf.log(pY))

#opt = tf.train.AdamOptimizer(0.01).minimize(cost_fn)

opt = tf.train.GradientDescentOptimizer(0.01).minimize(cost_fn)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

num_epochs = 40
for i in range(num_epochs):
  sess.run(opt, feed_dict={X:input_x, Y:output_x})


accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pY, 1), tf.argmax(Y, 1)), "float"))
accuracy_value = sess.run(accuracy, feed_dict={X:input_y, Y:output_y})

print(accuracy_value)

