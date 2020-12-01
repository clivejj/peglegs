import pickle


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
