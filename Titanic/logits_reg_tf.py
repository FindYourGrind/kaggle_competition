import titanic_preprocessor as preproc
import numpy as np
import tensorflow as tf

train_X, test_X, train_Y, test_Y = preproc.get_train_data(0.2)

train_Y_new = np.ndarray((train_Y.shape[0], 2))
test_Y_new = np.ndarray((test_Y.shape[0], 2))

for i in range(train_Y.shape[0]):
    train_Y_new[i] = [0, 1] if train_Y[i] else [1, 0]

train_Y = train_Y_new.astype(int)


for i in range(test_Y.shape[0]):
    test_Y_new[i] = [0, 1] if test_Y[i] else [1, 0]

test_Y = test_Y_new.astype(int)

num_features = train_X.shape[1]
num_classes = train_Y.shape[1]

X = tf.placeholder("float", [None, num_features])
Y = tf.placeholder("float", [None, num_classes])

W = tf.Variable(tf.zeros([num_features, num_classes]))
B = tf.Variable(tf.zeros([num_classes]))

pY = tf.nn.softplus(tf.matmul(X, W) + B)

cost_fn = -tf.reduce_sum(Y * tf.log(pY))

opt = tf.train.AdamOptimizer(0.01).minimize(cost_fn)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    num_epochs = 1000

    for i in range(num_epochs):
        sess.run(opt, feed_dict={X: train_X, Y: train_Y})

    print(pY)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pY, 1), tf.argmax(Y, 1)), "float"))

    accuracy_value = sess.run(accuracy, feed_dict={X: test_X, Y: test_Y})

    print(accuracy_value)