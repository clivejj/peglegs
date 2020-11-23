import gensim.downloader as gd

def get_vec():
	word_2_vec = gd.load('word2vec-google-news-300')
	return word_2_vec
	