import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from utils import unpickle, get_batch, setup
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
            acc = []

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
                acc += [sentiment_logit]
            batch_loss = (emotion_batch_loss / 8) + sentiment_batch_loss

        acc = np.squeeze(
            tf.cast(tf.math.sigmoid(tf.convert_to_tensor(acc)) > 0.5, np.float32), 2
        )
        accuracy = tf.reduce_sum(tf.cast(acc == batch_sentiment_labels, tf.int32))
        print("Batch Accuracy", accuracy / 64)
        print("Batch Loss", batch_loss)
        # batch_loss = emotion_batch_loss + sentiment_batch_loss

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
        print("Tweet", index)
        emotion_logits, sentiment_logits = model.call(
            tweet, embeddings, synonym_indices,
        )

        EmotionTP = tf.cast(tf.math.count_nonzero(emotion_logits * emotion_labels[index]), tf.float32)
        print("EmotionTP", EmotionTP)
        EmotionTN = tf.cast(tf.math.count_nonzero((emotion_labels[index] - 1) * (emotion_logits - 1)), tf.float32)
        print("EmotionTN", EmotionTN)
        EmotionFP = tf.cast(tf.math.count_nonzero(emotion_logits * (emotion_labels[index] - 1)), tf.float32)
        print("EmotionFP", EmotionFP)
        EmotionFN = tf.cast(tf.math.count_nonzero((emotion_logits - 1) * emotion_labels[index]), tf.float32)
        print("EmotionFN", EmotionFN)

        ePrecision = tf.math.divide_no_nan(EmotionTP, EmotionTP + EmotionFP)
        print("ePrecision", ePrecision)
        eRecall = tf.math.divide_no_nan(EmotionTP, EmotionTP + EmotionFN)
        print("eRecall", eRecall)
        
        emotionF1 += tf.math.divide_no_nan(2 * eRecall * ePrecision, eRecall + ePrecision)
        print("emotionF1", emotionF1)
        emotionRecall += eRecall
        print("emotionRecall", emotionRecall)

        SentimentTP = tf.cast(tf.math.count_nonzero(sentiment_logits * sentiment_labels[index]), tf.float32)
        print("SentimentTP", SentimentTP)
        SentimentTN = tf.cast(tf.math.count_nonzero((sentiment_logits - 1) * (sentiment_labels[index] - 1)), tf.float32)
        print("SentimentTN", SentimentTN)
        SentimentFP = tf.cast(tf.math.count_nonzero(sentiment_logits * (sentiment_labels[index] - 1)), tf.float32)
        print("SentimentFP", SentimentFP)
        SentimentFN = tf.cast(tf.math.count_nonzero((sentiment_logits - 1) * sentiment_labels[index]), tf.float32)
        print("SentimentFN", SentimentFN)

        sPrecision = tf.math.divide_no_nan(SentimentTP, SentimentTP + SentimentFP)
        print("sPrecision", sPrecision)
        sRecall = tf.math.divide_no_nan(SentimentTP, SentimentTP + SentimentFN)
        print("sRecall", sRecall)

        sentimentF1 += tf.math.divide_no_nan(2 * sRecall * sPrecision, sRecall + sPrecision)
        print("sentimentF1", sentimentF1)
        sentimentPrecision += sPrecision
        print("sentimentPrecision", sentimentPrecision)

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
    setup(overwrite=False)
    data = unpickle("data/data.pickle")
    # A dictionary that keys every word in our vocabulary to an index
    vocab = data[0]
    # A list of the tweets that we will be training on (2914 tweets)
    train_sentences = data[1]
    for i in range(len(train_sentences)):
        train_sentences[i] = tf.convert_to_tensor(train_sentences[i], tf.int32)
    # print("Sentences", len(sentences))
    # An embedding matrix that maps each word to a 300 Dimensional Embedding
    embeddings = tf.convert_to_tensor(data[2], tf.float32)
    # A dictionary that maps the index of a word to a list containing the indices of its 4 synonyms
    synonym_indices = tf.convert_to_tensor(data[3], tf.int32)

    # A list of sentiment labels corresponding to tweets; labels can be -1 (negative), 0 (objective), or (1) positive
    # (2914, 1)
    train_sentiment_labels = tf.convert_to_tensor(data[4], tf.float32)
    # A list of emotion labels corresponding to tweets; each label has 8 slots, where a 1 in that position corresponds to that
    # emotion being labelled. So, each tweet can be associated to several different emotions
    # Shape (2914, 8)
    train_emotion_labels = tf.convert_to_tensor(data[5], tf.float32)

    test_sentences = data[6]
    for j in range(len(test_sentences)):
        test_sentences[j] = tf.convert_to_tensor(test_sentences[j], tf.int32)

    test_sentiment_labels = tf.convert_to_tensor(data[7], tf.float32)

    test_emotion_labels = tf.convert_to_tensor(data[8], tf.float32)

    model = Model()

    train(
        model, train_sentences, train_emotion_labels, train_sentiment_labels, embeddings, synonym_indices
    )

    test(
        model, test_sentences, test_emotion_labels, test_sentiment_labels, embeddings, synonym_indices
    )


if __name__ == "__main__":
    main()
