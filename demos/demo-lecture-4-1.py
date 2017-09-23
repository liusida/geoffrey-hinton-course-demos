"""
This demo is based on Lecture 4
"A simple example of relational information "

Experience:
The dataset is not so properiate for typical neural network, because typical neural network want to learn some functions Y = f(X).
But in the dataset, we can see there are uncles and aunts relationships which are multipy-to-mulitpy.
So that the final accuracy can not be 1.0.
e.g.
When I ask "who is Colin's uncle?", the network only have one answer: Charles or Arthur, it cannot give us both answers..
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

# main relationship dataset
kinship_path = os.path.join( context.project_root, 'data/FamilyRelationship/kinship.txt' )
# I create this dataset only for showing the proper order of the names in the pop-window.
dic_person_path = os.path.join( context.project_root, 'data/FamilyRelationship/dic_person.txt' )

# prepare dataset
def prepare_dataset():

    # TODO: Can we do this task below more decently??

    # 1. prepare dic_persons
    with open(dic_person_path, 'r') as f:
        buff = f.read().lower()
    match = re.findall('[0-9]+\s+([a-z]+)', buff)
    match = np.array(match)
    dic_persons = dictionary.dictionary()
    for name in match:
        dic_persons.lookup(name) #Write names into dictionary
    print(dic_persons)
    dic_persons.freeze()

    # 2. read relationship dataset
    with open(kinship_path, 'r') as f:
        buff = f.read().lower()
    # pattern: father(Christopher, Arthur)
    match = re.findall('([a-z]+)[^a-z]*\([^a-z]*([a-z]+)[^a-z]*([a-z]+)[^a-z]*\)\n', buff)
    # Turning a word dataset into a vector dataset.
    match = np.array(match)
    dic_relations = dictionary.dictionary()
    relation = [ dic_relations.lookup(word) for word in match[:,0] ]
    # Here I changed the order to be as same as the slides
    # (colin has-father james)
    # person1 relationship person2
    person1 = [ dic_persons.lookup(word) for word in match[:,2] ]
    person2 = [ dic_persons.lookup(word) for word in match[:,1] ]
    dic_relations.freeze()

    # The local encoding mentioned in the Lecture, is also called one-hot encoding.
    def local_encoding(values, num_words):
        v = np.zeros([len(values), num_words])
        for i, val in enumerate(values):
            v[i, val] = 1
        return v

    relation_v = local_encoding(relation, 12)
    person1_v = local_encoding(person1, 24)
    person2_v = local_encoding(person2, 24)

    # TODO: End

    return person1_v, relation_v, person2_v, dic_persons, dic_relations

# Organize computational graphic
# [m,24] @ [24,6] + [1,6] => [m,6] concat => [m,12] @ [12,u] + [1,u] => [m,u] => [m,u] @ [u,6] + [1,6] => [m,6] @ [6,24] + [1,24] => [m,24]
# Person1 -----w+b------> Z_person --concatenate--> Z2 --w+b------------> Z3 --sigmoid--> A3 --w+b----------> Z4 --w+b-----------------> Z5(Y_hat)
# Relationship --w+b--> Z_relationship --^
# [m,12] * [24,6] + [1,6] => [m,6]
def define_graphic(learning_rate, units_in_hidden_layer):
    Person1 = tf.placeholder(dtype=tf.float32, shape=[None, 24])
    Relationship = tf.placeholder(dtype=tf.float32, shape=[None, 12])
    Person2 = tf.placeholder(dtype=tf.float32, shape=[None, 24])

    def _weight(a, b):
        return tf.Variable(tf.random_uniform(shape=[a, b], minval=-0.1, maxval=0.1, seed=1))
    weights = {
        'person_to_embedding': _weight(24,6),
        'relationship_to_embedding': _weight(12,6),
        'embedding_to_hidden': _weight(12,units_in_hidden_layer),
        'hidden_to_embedding': _weight(units_in_hidden_layer,6),
        'embedding_to_output': _weight(6,24)
    }
    def _bias(b):
        return tf.Variable(tf.zeros(shape=[1, b]))
    bias = {
        'person_to_embedding': _bias(6),
        'relationship_to_embedding': _bias(6),
        'embedding_to_hidden': _bias(units_in_hidden_layer),
        'hidden_to_embedding': _bias(6),
        'embedding_to_output': _bias(24)
    }
    for key, value in weights.items():
        tf.summary.histogram('weight: %s'%key, value)

    # @ is the operator for matmul
    Z_person = Person1 @ weights['person_to_embedding'] + bias['person_to_embedding']
    Z_relationship = Relationship @ weights['relationship_to_embedding'] + bias['relationship_to_embedding']
    Z2 = tf.concat((Z_person, Z_relationship), axis=1)

    Z3 = Z2 @ weights['embedding_to_hidden'] + bias['embedding_to_hidden']
    A3 = tf.nn.sigmoid(Z3)
    Z4 = A3 @ weights['hidden_to_embedding'] + bias['hidden_to_embedding']
    Z5 = Z4 @ weights['embedding_to_output'] + bias['embedding_to_output']

    Y_hat = Z5

    Prediction = tf.argmax(Y_hat, axis=1)
    Truth = tf.argmax(Person2, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Prediction, Truth), tf.float32))

    cost = tf.nn.softmax_cross_entropy_with_logits(logits=Y_hat, labels=Person2)
    loss = tf.reduce_mean(cost)

    # Adam is way more fast than normal Gradient Descent, so to save training time, I choose this.
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    return Person1, Relationship, Person2, Prediction, accuracy, loss, train, weights, bias, merged

# The training process
def fit(person1, relationship, person2, learning_rate=0.01, units_in_hidden_layer=4, iteration=100, ax=None, batch_size=None):
    Person1, Relationship, Person2, Prediction, accuracy, loss, train, weights, bias, merged \
        = define_graphic(learning_rate, units_in_hidden_layer)

    summary_writer = tf.summary.FileWriter(os.path.join(context.project_root,'tensorboard_log/',str(time.time())))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    num_total = person1.shape[0]
    if batch_size is None: # Full-batch
        batch_size = num_total
    for i in range(iteration):
        current_start = i * batch_size % num_total
        current_end = (i+1) * batch_size % num_total
        if current_end<=current_start:
            current_end = num_total
        #print(current_start, ":", current_end) # to check if I cut at the proper edges...

        batch_person1 = person1[current_start:current_end, :]
        batch_relationship = relationship[current_start:current_end, :]
        batch_person2 = person2[current_start:current_end, :]

        val_loss, val_accu, summary, val_prediction, _ \
            = sess.run([loss, accuracy, merged, Prediction, train], feed_dict={Person1: batch_person1, Relationship: batch_relationship, Person2: batch_person2})
        summary_writer.add_summary(summary, i)
        # Output 10 steps in total.
        if i*10%iteration==0 or i==iteration-1:
            val_loss, val_accu = sess.run([loss, accuracy], feed_dict={Person1: person1, Relationship: relationship, Person2: person2})
            print(round(i*10/iteration),"/ 10 -> loss: ", val_loss, ", accuracy: ", val_accu)
            # print("Prediction:", val_prediction)
            # print("Truth:", np.argmax(person2, axis=1) )
        # Update the image every 100 iterations.
        if i%100==0 or i==iteration-1:
            if ax is not None:
                person_to_embedding = sess.run(weights['person_to_embedding'])
                for ind_i in range(6):
                    for ind_j in range(24):
                        single_weight = person_to_embedding[ind_j,ind_i]
                        if single_weight<=0:
                            color = 'black'
                            single_weight = -single_weight
                        else:
                            color = 'white'
                        show.show_box(ax, ind_i, ind_j, single_weight, color)
                ax.set_title('step: %d'%i)
                ax.get_figure().canvas.draw()
                #time.sleep(0.1)

    # Finally report the wrong predictions to analysis.
    val_prediction, = sess.run([Prediction], feed_dict={Person1: person1, Relationship: relationship, Person2: person2})
    wrong_predictions = np.not_equal(val_prediction, np.argmax(person2, axis=1))
    wrongs = {
        'person1': np.argmax(person1, axis=1)[wrong_predictions],
        'relationship': np.argmax(relationship, axis=1)[wrong_predictions],
        'person2': np.argmax(person2, axis=1)[wrong_predictions]
    }
    sess.close()
    return wrongs


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

    # Hint: I have tuned to show a better result:
    wrongs = fit(a, b, c, iteration=3000, ax=ax, learning_rate=0.001, units_in_hidden_layer=12, batch_size=None)

    print("\nAfter training, we still have those wrong predictions:")
    for key,value in enumerate(wrongs['person1']):
        print( dic_persons.lookup_index(wrongs['person1'][key]),
               dic_relations.lookup_index(wrongs['relationship'][key]),
               dic_persons.lookup_index(wrongs['person2'][key])
               )

    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()

# ----------------------------------------------------------------------------
# [1] If you'd like to implement using numpy, take a look at:
# https://github.com/radekosmulski/10_neural_nets/blob/master/family_tree.py