import csv
from ekphrasis.classes.preprocessor import TextPreProcessor
import numpy as np
import gensim.downloader as gd
from os import path
import pickle
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.tokenizer import SocialTokenizer
import gensim


seg = Segmenter(corpus="twitter")

text_processor = TextPreProcessor(
    normalize=["url", "number", "user"],
    segmenter="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
)


def preprocess(sentimentFile, emotionFile, word_2_vec):
    # dict with word (str) -> word index (int)
    vocab = {}
    word_counter = 0
    with open(sentimentFile, newline="", encoding="latin-1") as f:
        # remove first line
        data = list(csv.reader(f))[1:]
    num_tweets = len(data)
    # (num_tweets x 1) aray containing sentiment labels for each sentence (0, 1)
    sentiment_labels = []
    # list of np arrays, with each array containing indices for the words in that sentence
    sentences = []
    i = 0
    # loop thru every row in csv file, extract data
    for tweet_index in range(num_tweets):
        row = data[tweet_index]
        if row[4] == "other":
            continue
        # extract text of tweet and clean it with ekphrasis
        tweet = text_processor.pre_process_doc(row[0])
        # tweet = sum([seg.segment(x).split() for x in tweet if len(x) > 1], [])
        tweet = sum([seg.segment(x).split() for x in tweet], [])

        # create array for each tweet holding indices corresponding to words
        word_indices = []
        for word in tweet:
            # if word is not new, simply add its index to list
            if word in vocab:
                word_indices.append(vocab[word])
            # if word is new, add it to vocab and then add its index to list
            elif word in word_2_vec:
                vocab[word] = word_counter
                word_indices.append(word_counter)
                word_counter += 1
            else:
                i += 1

        sentences += [np.asarray(word_indices, dtype=np.int32)]

        # extract sentiment
        sentiment = row[4]
        sentiment = ["neg", "pos"].index(sentiment)
        sentiment_labels += [sentiment]
    sentiment_labels = np.reshape(np.asarray(sentiment_labels), (-1, 1))
    # print(np.shape(sentiment_labels))
    # create labels for emotions for each sentence
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

    return (vocab, sentences, sentiment_labels, emotion_labels)


def expand_vocab(vocab, word_2_vec, embeddings, display_progress=True):
    corpus = list(vocab.keys())
    len_corpus = len(vocab)
    synonym_indices = np.zeros((len_corpus, 4))
    if display_progress:
        percent_progress = -1
    i = len(vocab)
    for index, word in enumerate(corpus):
        if display_progress:
            if np.floor(index * 100 / len_corpus) != percent_progress:
                percent_progress = np.floor(index * 100 / len_corpus)
                print("approx %d%% complete" % percent_progress)
        synonyms = [
            x[0] for x in word_2_vec.similar_by_vector(embeddings[vocab[word]], topn=4)
        ]
        indices = []
        for synonym in synonyms:
            if synonym in vocab:
                indices += [vocab[synonym]]
            else:
                vocab[synonym] = i
                indices += [i]
                i += 1
        synonym_indices[index, :] = np.asarray(indices, dtype=np.int32)
    return synonym_indices


def create_embeddings(vocab, word_2_vec):
    embeddings = np.zeros((len(vocab), 300))
    for word in vocab:
        embeddings[vocab[word], :] = word_2_vec[word]
    return embeddings


def get_vec(limit=None):
    targetFile = "./data/word2vec-google-news-300/word2vec-google-news-300.gz"
    if not path.exists(targetFile):
        gensim.downloader.BASE_DIR = "./data"
        gensim.downloader.load("word2vec-google-news-300")
    return gensim.models.KeyedVectors.load_word2vec_format(
        targetFile, binary=True, limit=limit
    )

