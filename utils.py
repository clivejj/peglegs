import pickle
import tensorflow as tf
from preprocessing import preprocess, expand_vocab, create_embeddings, get_vec
import gensim
from os import path


def unpickle(file):
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    return d


"""
    Retrieves a batch of inputs, emotion labels, and sentiment labels
"""


def get_batch(train_inputs, emotion_labels, sentiment_labels, batch_num, batch_size):
    batch_inputs = train_inputs[
        (batch_size * batch_num) : (batch_size * batch_num) + batch_size
    ]
    batch_emotion_labels = emotion_labels[
        (batch_size * batch_num) : (batch_size * batch_num) + batch_size
    ]
    batch_sentiment_labels = sentiment_labels[
        (batch_size * batch_num) : (batch_size * batch_num) + batch_size
    ]

    return batch_inputs, batch_emotion_labels, batch_sentiment_labels


def construct_row(
    index, word, dense_out, hidden_states, embedding_matrix, synonym_indices
):
    synonym_embeddings = tf.nn.embedding_lookup(embedding_matrix, synonym_indices[word])
    out_temp = tf.expand_dims(dense_out[index, :], 1)
    coefficients = tf.math.exp(tf.matmul(synonym_embeddings, out_temp))
    m = tf.reduce_sum(coefficients * synonym_embeddings, 0)
    h = tf.reshape(m + hidden_states[index, :], (1, -1))
    return h


def setup(overwrite=False):
    if path.exists("data/data.pickle") and not overwrite:
        return

    word_2_vec = get_vec(10 ** 6)
    print("done loading word_2_vec")

    vocab, train_sentences, train_sentiment_labels, train_emotion_labels = preprocess(
        "data/train_tweet_sentiment.csv", "data/train_emotion.csv", word_2_vec
    )
    print("done first pre-process")

    ____, test_sentences, test_sentiment_labels, test_emotion_labels = preprocess(
        "data/test_tweet_sentiment.csv", "data/test_emotion.csv", word_2_vec
    )

    temp_embeddings = create_embeddings(vocab, word_2_vec)
    word_2_vec = get_vec(5 * (10 ** 5))
    synonym_indices = expand_vocab(vocab, word_2_vec, temp_embeddings)
    print("done vocab expansion")

    temp_embeddings = None
    word_2_vec = get_vec(10 ** 6)

    embeddings = create_embeddings(vocab, word_2_vec)
    word_2_vec = None

    data = (
        vocab,
        train_sentences,
        embeddings,
        synonym_indices,
        train_sentiment_labels,
        train_emotion_labels,
        test_sentences,
        test_sentiment_labels,
        test_emotion_labels,
    )

    pickle.dump(data, open("data/data.pickle", "wb"))


"""def f1(predictions, actual):
    predict_pos = predictions == 1
    actual_pos = actual == 1
    TP = tf.reduce_sum(tf.cast(predict_pos and actual_pos, tf.float32))
    FP = tf.reduce_sum(tf.cast(predict_pos and (actual == 0), tf.float32))
    FN = tf.reduce_sum(tf.cast((predictions == 0) and actual_pos, tf.float32))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return ((1 / precision) + (1 / recall)) / 2"""

