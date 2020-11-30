from preprocessing import preprocess, expand_vocab, create_embeddings, get_vec
import gensim
import pickle

# word_2_vec = get_vec()

word_2_vec = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True
)

print("done loading vec")

vocab, sentences, sentiment_labels, emotion_labels = preprocess(
    "data/train_tweet_sentiment.csv", "data/train_emotion.csv", word_2_vec
)
print("done pre-proc")
print(len(vocab))


ind = expand_vocab(vocab, word_2_vec)
print("done expand")


embeddings = create_embeddings(vocab, word_2_vec)

data = [vocab, sentences, embeddings, ind]

pickle.dump(data, open("data.pickle", "wb"))
