import csv
from ekphrasis.classes.preprocessor import TextPreProcessor
import numpy as np

text_processor = TextPreProcessor(
    normalize=["url", "number", "user"],
    segmenter="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
)


def preprocess(sentimentFile, emotionFile):
    # dict with word (str) -> word index (int)
    vocab = {}
    word_counter = 0
    with open(sentimentFile, newline="", encoding="latin-1") as f:
        # remove first line
        data = list(csv.reader(f))[1:]
    num_tweets = len(data)
    # (num_tweets x 1) aray containing sentiment labels for each sentance (-1, 0, 1)
    sentiment_labels = np.zeros((num_tweets, 1))
    # list of np arrays, with each array containing indices for the words in that sentance
    sentances = [None] * num_tweets
    # loop thru every row in csv file, extract data
    for tweet_index in range(num_tweets):
        row = data[tweet_index]
        # extract text of tweet and clean it with ekphrasis
        tweet = text_processor.pre_process_doc(row[0]).split()
        # create array for each tweet holding indices corresponding to words
        num_words = len(tweet)
        word_indices = np.zeros((num_words, 1), dtype=np.int32)
        for i in range(num_words):
            word = tweet[i]
            # if word is not new, simply add its index to list
            if word in vocab:
                word_indices[i] = vocab[word]
            # if word is new, add it to vocab and then add its index to list
            else:
                vocab[word] = word_counter
                word_indices[i] = word_counter
                word_counter += 1

        sentances[tweet_index] = word_indices

        # extract sentiment
        sentiment = row[4]
        sentiment = ["neg", "other", "pos"].index(sentiment) - 1
        sentiment_labels[tweet_index] = sentiment

    # create labels for emotions for each sentance
    # label is either 0 or 1 for each emotion
    emotion_labels = np.zeros((num_tweets, 8))
    # loop thru new emotion file
    with open(emotionFile, newline="", encoding="latin-1") as f:
        data = list(csv.reader(f, delimiter="\t"))
    if len(data) != num_tweets:
        raise ValueError("Both files must contain same number of tweets")
    for tweet_index in range(num_tweets):
        row = (np.asarray(data[tweet_index]) != "---").astype(np.float32)
        # this line is necessary because one row in csv had no emotions and therefore was formatted incorrectly
        if len(row) == 1:
            row = np.zeros((1, 8))
        emotion_labels[tweet_index, :] = row

    return (vocab, sentances, sentiment_labels, emotion_labels)


preprocess("data/train_tweet_sentiment.csv", "data/train_emotion.csv")

