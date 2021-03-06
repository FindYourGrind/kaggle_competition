import titanic_preprocessor as preproc
import tensorflow as tf
# Build Neural Network
from collections import namedtuple

train_x, valid_x, train_y, valid_y = preproc.get_train_data(0.2)


def build_neural_network(hidden_units=10):
    tf.reset_default_graph()

    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])

    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.Variable(True, dtype=tf.bool)

    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs, hidden_units, activation=None, kernel_initializer=initializer)
    fc = tf.layers.batch_normalization(fc, training=is_training)
    fc = tf.nn.relu(fc)

    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes
    export_nodes = ['inputs', 'labels', 'learning_rate', 'is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


model = build_neural_network()


def get_batch(data_x, data_y, batch_size=32):
    batch_n = len(data_x) // batch_size
    for i in range(batch_n):
        batch_x = data_x[i * batch_size:(i + 1) * batch_size]
        batch_y = data_y[i * batch_size:(i + 1) * batch_size]

        yield batch_x, batch_y


epochs = 200
train_collect = 50
train_print = train_collect * 2

learning_rate_value = 0.01
batch_size = 16

x_collect = []
train_loss_collect = []
train_acc_collect = []
valid_loss_collect = []
valid_acc_collect = []

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 0
    for e in range(epochs):
        for batch_x, batch_y in get_batch(train_x, train_y, batch_size):
            iteration += 1
            feed = {model.inputs: train_x,
                    model.labels: train_y,
                    model.learning_rate: learning_rate_value,
                    model.is_training: True
                    }

            train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)

            if iteration % train_collect == 0:
                x_collect.append(e)
                train_loss_collect.append(train_loss)
                train_acc_collect.append(train_acc)

                if iteration % train_print == 0:
                    print("Epoch: {}/{}".format(e + 1, epochs),
                          "Train Loss: {:.4f}".format(train_loss),
                          "Train Acc: {:.4f}".format(train_acc))

                feed = {model.inputs: valid_x,
                        model.labels: valid_y,
                        model.is_training: False
                        }
                val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                valid_loss_collect.append(val_loss)
                valid_acc_collect.append(val_acc)

                if iteration % train_print == 0:
                    print("Epoch: {}/{}".format(e + 1, epochs),
                          "Validation Loss: {:.4f}".format(val_loss),
                          "Validation Acc: {:.4f}".format(val_acc))

    print(model.predicted)
    saver.save(sess, "./titanic.ckpt")
