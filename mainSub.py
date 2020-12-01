
from preprocessing import preprocess, expand_vocab, create_embeddings, get_vec
import gensim
import pickle

word_2_vec = get_vec(10 ** 6)

print("done loading vec")

vocab, sentences, sentiment_labels, emotion_labels = preprocess(
    "data/train_tweet_sentiment.csv", "data/train_emotion.csv", word_2_vec
)
print("done first pre-process")


temp_embeddings = create_embeddings(vocab, word_2_vec)
word_2_vec = get_vec(5 * (10 ** 5))
print("done loading vec for vocab expansion")
synonym_indices = expand_vocab(vocab, word_2_vec, temp_embeddings)
print("done vocab expansion")


temp_embeddings = None
word_2_vec = get_vec(10 ** 6)


embeddings = create_embeddings(vocab, word_2_vec)

data = [vocab, sentences, embeddings, synonym_indices, sentiment_labels, emotion_labels]

pickle.dump(data, open("data.pickle", "wb"))
