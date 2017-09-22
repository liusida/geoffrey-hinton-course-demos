"""
This demo is based on Lecture 4
"A simple example of relational information "

"""
import os, re, time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# My Useful Tools
import helper.dictionary as dictionary
import helper.show_tools
show = helper.show_tools.lec4()
import context

dataset_path = os.path.join( context.project_root, 'data/FamilyRelationship/kinship.txt' )
# prepare dataset
def prepare_dataset():
    with open(dataset_path, 'r') as f:
        buff = f.read().lower()

    # pattern: father(Christopher, Arthur)
    match = re.findall('([a-z]+)[^a-z]*\([^a-z]*([a-z]+)[^a-z]*([a-z]+)[^a-z]*\)\n', buff)

    # TODO: Can we do this task below more decently??
    # Turning a word dataset into a vector dataset.
    match = np.array(match)

    dic_persons = dictionary.dictionary()
    dic_relations = dictionary.dictionary()
    relation = [ dic_relations.lookup(word) for word in match[:,0] ]
    person1 = [ dic_persons.lookup(word) for word in match[:,1] ]
    person2 = [ dic_persons.lookup(word) for word in match[:,2] ]
    dic_persons.freeze()
    dic_relations.freeze()

    def local_encoding(values, num_words):
        v = np.zeros([len(values), num_words])
        for i in values:
            v[i, values[i]] = 1
        return np.array(v)

    relation_v = local_encoding(relation, 12)
    person1_v = local_encoding(person1, 24)
    person2_v = local_encoding(person2, 24)

    # TODO: End

    return person1_v, relation_v, person2_v, dic_persons, dic_relations

# computational graphic
# TODO: This model can not be well trained yet!
# Person1 ---------> Z_person -------+--> Z2 ----> Y_hat(Z3)
# Relationship --> Z_relationship --^

def define_graphic(learning_rate):
    Person1 = tf.placeholder(dtype=tf.float32, shape=[None, 24])
    Relationship = tf.placeholder(dtype=tf.float32, shape=[None, 12])
    Person2 = tf.placeholder(dtype=tf.float32, shape=[None, 24])

    distributed_encoding_of_person1 = tf.Variable(tf.random_normal(shape=[24, 6]))
    distributed_encoding_of_relationship = tf.Variable(tf.random_normal(shape=[12, 6]))
    bias_1 = tf.Variable(tf.zeros(shape=[1,6]))
    bias_2 = tf.Variable(tf.zeros(shape=[1,6]))

    distributed_encoding_of_person2 = tf.Variable(tf.random_normal(shape=[12, 24]))
    # @ is the operator for matmul
    Z_person = Person1 @ distributed_encoding_of_person1 + bias_1
    Z_relationship = Relationship @ distributed_encoding_of_relationship + bias_2
    Z2 = tf.concat((Z_person, Z_relationship), axis=1)
    Z3 = Z2 @ distributed_encoding_of_person2
    Y_hat = Z3

    Prediction = tf.argmax(Y_hat)
    Truth = tf.argmax(Person2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Prediction, Truth), tf.float32))
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=Y_hat, labels=Person2)
    loss = tf.reduce_mean(cost)
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    return Person1, Relationship, Person2, Prediction, accuracy, loss, train, distributed_encoding_of_person1, distributed_encoding_of_relationship, distributed_encoding_of_person2, merged

def fit(person1, relationship, person2, learning_rate=0.01, iteration=100, ax=None):
    summary_writer = tf.summary.FileWriter(os.path.join(context.project_root,'tensorboard_log/',str(time.time())))
    Person1, Relationship, Person2, Prediction, accuracy, loss, train, w1, w2, w3, merged \
        = define_graphic(learning_rate)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(iteration):
        val_loss, val_accu, summary, _ = sess.run([loss, accuracy, merged, train], feed_dict={Person1: person1, Relationship: relationship, Person2: person2})
        summary_writer.add_summary(summary, i)
        if i*10%iteration==0:
            print((i*10/iteration),"/ 10 -> loss: ", val_loss, ", accuracy: ", val_accu)
            if ax is not None:
                weights = sess.run([w1, w2, w3])
                for ind_i in range(6):
                    for ind_j in range(24):
                        weight = weights[0][ind_j,ind_i]
                        if weight<=0:
                            color = 'black'
                            weight = -weight
                        else:
                            color = 'white'
                        show.show_box(ax, ind_i, ind_j, weight, color)
                ax.set_title(i)
                ax.get_figure().canvas.draw()
                time.sleep(0.1)

    val_prediction, = sess.run([Prediction], feed_dict={Person1: person1, Relationship: relationship, Person2: person2})
    sess.close()
    return val_prediction


def main():
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=[10,5])
    ax = fig.gca()
    ax.set_xlim(0,100)
    ax.set_ylim(0,50)
    show.show_background(ax)

    a, b, c, dic_persons, dic_relations = prepare_dataset()
    for i in range(12):
        show.show_name(ax, i, dic_persons.lookup_index(i))

    val_prediction = fit(a, b, c, iteration=1000, ax=ax, learning_rate=0.01)
    print(val_prediction)
    for i in range(val_prediction.shape[0]):
        predict_name = dic_persons.lookup_index(val_prediction[i])
        #print(predict_name)

    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()

