import tensorflow as tf
import numpy as np
from utils import construct_row


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # HyperParamaters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.embedding_size = 300
        self.h1 = 300
        self.batch_size = 64
        # Trainable parameters

        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=int(self.h1 / 2), return_sequences=True,)
        )
        self.primary_attention_dense_layer = tf.keras.layers.Dense(self.h1)
        self.secondary_attention_dense_layer = tf.keras.layers.Dense(
            1, activation="tanh"
        )
        self.emotion_output_layer = tf.keras.layers.Dense(units=8)
        self.sentiment_output_layer = tf.keras.layers.Dense(units=1)

    def call(self, sentence, embedding_matrix, synonym_indices):

        embeddings = tf.nn.embedding_lookup(embedding_matrix, sentence)
        # print("Embeddings Shape", np.shape(embeddings))
        hidden_states = tf.squeeze(self.biLSTM(tf.expand_dims(embeddings, 0)))
        # print(hidden_states)
        # print("Hidden states shape", np.shape(hidden_states))
        h_hats = self.primary_attention(
            sentence, hidden_states, embedding_matrix, synonym_indices
        )
        # print("h_hats Shape post call", np.shape(h_hats))

        h_bars = h_hats

        H_HAT = tf.expand_dims(self.secondary_attention(h_hats), 0)

        H_BAR = H_HAT
        # print("H_HAT Shape", np.shape(H_HAT))

        emotion_logits = self.emotion_output_layer(H_BAR)
        # print("Emotion Logits Shape", np.shape(emotion_logits))
        # emotion_logits = tf.where(emotion_logits > .5, 1.0, 0.0)
        # print("Emotion Logits", emotion_logits)

        sentiment_logits = self.sentiment_output_layer(H_HAT)
        # sentiment_logits = tf.where(sentiment_logits > .5, 1.0, -1.0)

        # return emotion_logits, sentiment_logit
        return emotion_logits, sentiment_logits

    def primary_attention(
        self, sentence, hidden_states, embedding_matrix, synonym_indices
    ):
        """
		TODO: 
		for each of the 4 synonyms of a word (retrieved using wn.synset),
		calculate its attention using tf.math.exp( tf.matmul(
		tf.matmul(tf.transpose(hidden_state for word), self.primary_attention) + self.primary_bias, synonym )

		Then, calculate m for the word by summing this up, and then create h by concatenating the
		hidden state for the word and m
		"""
        out = self.primary_attention_dense_layer(hidden_states)

        # construct first row of hs matrix
        hs = construct_row(
            0, sentence[0], out, hidden_states, embedding_matrix, synonym_indices
        )
        # construct remaining rows of hs matrix
        for index in range(1, len(sentence)):
            word = sentence[index]
            h = construct_row(
                index, word, out, hidden_states, embedding_matrix, synonym_indices
            )
            hs = tf.concat([hs, h], 0)

        return hs

    def secondary_attention(self, h_hats):
        """
		TODO:
		for each of the words in a sentence,
		calculate its attention using tf.math.exp(tf.math.tanh(
		tf.matmul(tf.transpose(hidden_state for word), self.secondary_attention) + self.secondary_bias))

		Then, calculate m for the word by summing this up, and then create h by concatenating the
		hidden state for the word and m
		"""
        coefficients = tf.math.exp(self.secondary_attention_dense_layer(h_hats))
        return tf.reduce_sum(coefficients * h_hats, 0)

    def loss_function(self, labels, logits):
        # Calculate the sum of the loss by comparing the labels with the inputted logits
        # print("Labels Type", labels)
        # print("Logits Type", logits)
        # return tf.convert_to_tensor(6.3)
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        )

