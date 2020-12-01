import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Model
from preprocessing import get_vec, unpickle
from model import Model

def train(model, train_inputs, emotion_labels, sentiment_labels):

    for batch_num in range(int(len(train_inputs)/model.batch_size)):

        batch_inputs, batch_emotion_labels, batch_sentiment_labels = get_batch(train_inputs, 
            emotion_labels, sentiment_labels, batch_num, model.batch_size)

        with tf.GradientTape() as tape:
            emotion_batch_loss = 0
            sentiment_batch_loss = 0

            for index, tweet in enumerate(batch_inputs):
                emotion_label, 
    # for batch_num in range(int(train_inputs.shape[0]/model.batch_size)):
    #     batch_inputs, batch_emotion_labels, batch_sentiment_labels = get_batch(train_inputs, 
    #         emotion_labels, sentiment_labels, batch_num, model.batch_size)

    #     with tf.GradientTape() as tape:
    #         emotion_logits, sentiment_logits = model.call(batch_inputs, None)
    #         emotion_batch_loss = model.loss_function(emotion_logits, batch_emotion_labels)
    #         sentiment_batch_loss = model.loss_function(sentiment_logits, batch_sentiment_labels)
    #         batch_loss = emotion_batch_loss + sentiment_batch_loss

    #         gradients = tape.gradient(batch_loss, model.trainable_parameters)
    #         model.optimizer.apply_gradients(zip(gradients, model.trainable_parameters))

    #         pass
def test(model, test_inputs, emotion_labels, sentiment_labels):

    test_inputs = np.array(test_inputs)
    emotion_labels = np.array(emotion_labels)
    sentiment_labels = np.array(sentiment_labels)

    loss = tf.Variable([0], dtype=tf.float32)

    for batch_num in range(int(train_inputs.shape[0]/model.batch_size)):
        batch_inputs, batch_emotion_labels, batch_sentiment_labels = get_batch(test_inputs,
                     emotion_labels, sentiment_labels, batch_num, model.batch_size)
        emotion_logits, sentiment_logits = model.call(batch_inputs, None)
        emotion_batch_loss = model.loss_function(emotion_logits, batch_emotion_labels)
        sentiment_batch_loss = model.loss_function(sentiment_logits, batch_sentiment_labels)
        loss = loss + batch_loss

    pass
    '''
    Calculate F1 score for emotion and sentiment, probably using tfa.metrics.F1Score
    and then print each

    Maybe make use of some visualization tool?

    '''

'''
    Retrieves a batch of inputs, emotion labels, and sentiment labels
'''
def get_batch(train_inputs, emotion_labels, sentiment_labels, batch_num, batch_size):
    batch_inputs = train_inputs[(batch_size * batch_num) : (batch_size * batch_num) + batch_size]
    batch_emotion_labels = emotion_labels[(batch_size * batch_num) : (batch_size * batch_num) + batch_size]
    batch_sentiment_labels = sentiment_labels[(batch_size * batch_num) : (batch_size * batch_num) + batch_size]

    return batch_inputs, batch_emotion_labels, batch_sentiment_labels

def main():
    #Unpickles data from file, stores it as dictionary of length 4
    data = unpickle('data.pickle')
    #A dictionary that keys every word in our vocabulary to an index
    vocab = data[0]
    #A list of the tweets that we will be training on (2914 tweets)
    sentences = data[1]
    # print("Sentences", len(sentences))
    #An embedding matrix that maps each word to a 300 Dimensional Embedding
    embeddings = data[2]
    #A dictionary that maps the index of a word to a list containing the indices of its 4 synonyms
    synonym_indices = data[3]
    #A list of sentiment labels corresponding to tweets; labels can be -1 (negative), 0 (objective), or (1) positive
    #(2914, 1)
    sentiment_labels = data[4]
    #A list of emotion labels corresponding to tweets; each label has 8 slots, where a 1 in that position corresponds to that
    #emotion being labelled. So, each tweet can be associated to several different emotions
    #Shape (2914, 8)
    emotion_labels = data[5]

    '''
        Splits the data into training and testing portions, as determined by the test_fraction parameter
    '''
    test_fraction = 0.1

    #Currently of length 2622
    training_sentences = sentences[ : int((1 - test_fraction) * len(sentences))]
    training_sentiment_labels = sentiment_labels[ : int((1 - test_fraction) * sentiment_labels.shape[0])]
    training_emotion_labels = emotion_labels[ : int((1 - test_fraction) * emotion_labels.shape[0])]


    #Currently of length 292
    testing_sentences = sentences[int((1 - test_fraction) * len(sentences)) :]
    testing_sentiment_labels = sentiment_labels[int((1 - test_fraction) * sentiment_labels.shape[0]) : ]
    testing_emotion_labels = emotion_labels[int((1 - test_fraction) * emotion_labels.shape[0]) : ]


    model = Model()

    train(model, training_sentences, training_emotion_labels, training_sentiment_labels)


 




if __name__ == '__main__':
    main()
