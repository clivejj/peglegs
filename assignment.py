import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from utils import unpickle, get_batch
from model import Model


def train(
    model, train_inputs, emotion_labels, sentiment_labels, embeddings, synonym_indices
):

    for batch_num in range(int(len(train_inputs) / model.batch_size)):

        print("Batch Number", batch_num)
        batch_inputs, batch_emotion_labels, batch_sentiment_labels = get_batch(
            train_inputs, emotion_labels, sentiment_labels, batch_num, model.batch_size
        )

        with tf.GradientTape() as tape:
            emotion_batch_loss = 0
            sentiment_batch_loss = 0
            # acc = []

            for index, tweet in enumerate(batch_inputs):
                emotion_logits, sentiment_logit = model.call(
                    tweet, embeddings, synonym_indices
                )
                emotion_batch_loss += model.loss_function(
                    tf.expand_dims(batch_emotion_labels[index], 0), emotion_logits,
                )
                sentiment_batch_loss += model.loss_function(
                    tf.expand_dims(batch_sentiment_labels[index], 0,), sentiment_logit,
                )
                # acc += [sentiment_logit]
            batch_loss = emotion_batch_loss + sentiment_batch_loss

        """acc = np.squeeze(
            tf.cast(tf.math.sigmoid(tf.convert_to_tensor(acc)) > 0.5, np.float32), 2
        )
        print(tf.cast(batch_sentiment_labels, np.int32))
        print(acc)"""
        print(batch_loss)
        # batch_loss = emotion_batch_loss + sentiment_batch_loss
        """print(sentiment_logit)
        print(emotion_logits)
        print(batch_sentiment_labels)
        print(accuracy)"""

        gradients = tape.gradient(batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(
    model, test_inputs, emotion_labels, sentiment_labels, embeddings, synonym_indices
):
    emotionF1 = 0
    emotionRecall = 0
    sentimentF1 = 0
    sentimentPrecision = 0

    for index, tweet in enumerate(test_inputs):
        emotion_logits, sentiment_logits = model.call(batch_inputs, None)

        eRecall = tf.compat.v1.metrics.recall(emotion_labels[index], emotion_logits)
        ePrecision = tf.compat.v1.metrics.precision(
            emotion_labels[index], emotion_logits
        )

        emotionF1 += 2 * (eRecall * ePrecision) / (eRecall + ePrecision)
        emotionRecall += eRecall

        sRecall = tf.compat.v1.metrics.recall(sentiment_labels[index], sentiment_logit)
        sPrecision = tf.compat.v1.metrics.recall(
            sentiment_labels[index], sentiment_logit
        )

        sentimentF1 += 2 * (sRecall * sPrecision) / (sRecall + sPrecision)
        sentimentPrecision += sPrecision

    average_emotion_F1 = emotionF1 / len(test_inputs)
    print("Average Emotion F1", average_emotion_F1)
    average_emotion_Recall = emotionRecall / len(test_inputs)
    print("Average Emotion Recall", average_emotion_Recall)
    average_sentiment_F1 = sentimentF1 / len(test_inputs)
    print("Average Sentiment F1", average_sentiment_F1)
    average_sentiment_Precision = sentimentPrecision / len(test_inputs)
    print("Average Sentiment Precision", average_sentiment_Precision)


def main():
    # Unpickles data from file, stores it as dictionary of length 4
    data = unpickle("data/data.pickle")
    # A dictionary that keys every word in our vocabulary to an index
    vocab = data[0]
    # A list of the tweets that we will be training on (2914 tweets)
    sentences = data[1]
    for i in range(len(sentences)):
        sentences[i] = tf.convert_to_tensor(sentences[i], tf.int32)
    # print("Sentences", len(sentences))
    # An embedding matrix that maps each word to a 300 Dimensional Embedding
    embeddings = tf.convert_to_tensor(data[2], tf.float32)
    # A dictionary that maps the index of a word to a list containing the indices of its 4 synonyms
    synonym_indices = tf.convert_to_tensor(data[3], tf.int32)

    # A list of sentiment labels corresponding to tweets; labels can be -1 (negative), 0 (objective), or (1) positive
    # (2914, 1)
    sentiment_labels = tf.convert_to_tensor(data[4], tf.float32)
    # A list of emotion labels corresponding to tweets; each label has 8 slots, where a 1 in that position corresponds to that
    # emotion being labelled. So, each tweet can be associated to several different emotions
    # Shape (2914, 8)
    emotion_labels = tf.convert_to_tensor(data[5], tf.float32)

    """
    Splits the data into training and testing portions, as determined by the test_fraction parameter
    test_fraction = 0.1

    #Currently of length 2622
    training_sentences = sentences[ : int((1 - test_fraction) * len(sentences))]
    training_sentiment_labels = sentiment_labels[ : int((1 - test_fraction) * sentiment_labels.shape[0])]
    training_emotion_labels = emotion_labels[ : int((1 - test_fraction) * emotion_labels.shape[0])]


    #Currently of length 292
    testing_sentences = sentences[int((1 - test_fraction) * len(sentences)) :]
    testing_sentiment_labels = sentiment_labels[int((1 - test_fraction) * sentiment_labels.shape[0]) : ]
    testing_emotion_labels = emotion_labels[int((1 - test_fraction) * emotion_labels.shape[0]) : ]"""

    model = Model()

    train(
        model, sentences, emotion_labels, sentiment_labels, embeddings, synonym_indices
    )


if __name__ == "__main__":
    main()
