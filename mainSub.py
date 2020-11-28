from preprocessing import preprocess, expand_vocab, create_embeddings, get_vec


vocab, sentences, sentiment_labels, emotion_labels = preprocess(
    "data/train_tweet_sentiment.csv", "data/train_emotion.csv"
)

vocab = expand_vocab(vocab)

word_2_vec = get_vec()

embeddings = create_embeddings(vocab, word_2_vec)

print(embeddings)
