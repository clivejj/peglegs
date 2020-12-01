import tensorflow as tf
import numpy as np


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
            tf.keras.layers.LSTM(units=int(self.h1 / 2), return_sequences=True),
            dtype=np.float32,
        )
        self.primary_attention_dense_layer = tf.keras.layers.Dense(self.h1)
        self.secondary_attention_dense_layer = tf.keras.layers.Dense(
            1, activation="tanh"
        )
        self.emotion_output_layer = tf.keras.layers.Dense(units=8, activation="sigmoid")
        self.sentiment_output_layer = tf.keras.layers.Dense(
            units=2, activation="sigmoid"
        )

    def call(self, batch_inputs, embedding_matrix, synonym_indices):
        for sentance in batch_inputs:
            # get embedding from word2vec embedding layer
            embeddings = tf.nn.embedding_lookup(embedding_matrix, sentance)
            hidden_states = tf.squeeze(self.biLSTM(tf.expand_dims(embeddings, 0)))
            h_hats = self.primary_attention(
                sentance, hidden_states, embedding_matrix, synonym_indices
            )
            H_HAT = self.secondary_attention(h_hats)
            print("KEEP IMPlEMENTING AFTER HERE")
            return

        """
		TODO: Pass in hidden state output of biLSTM layer into primary attention layer for emotion
		and for sentiment primary attention layers

		Then, pass in the output of each of these, which will be h bar and h cap respectively,
		into their respective secondary attention layers to get H bar and H cap respective

		Add in dropout!
		"""

        """h_bar = primary_attention((final_memory_output, final_carry_output))

        h_cap = primary_attention((final_memory_output, final_carry_output))

        H_bar = secondary_attention(h_bar)

        H_cap = secondary_attention(h_cap)

        emotion_logits = tf.nn.softmax(self.emotion_output_layer(H_bar))

        sentiment_logits = tf.nn.softmax(self.sentiment_output_layer(H_cap))

        return emotion_logits, sentiment_logits"""

    def primary_attention(
        self, sentance, hidden_states, embedding_matrix, synonym_indices
    ):
        """
		TODO: 
		for each of the 4 synonyms of a word (retrieved using wn.synset),
		calculate its attention using tf.math.exp( tf.matmul(
		tf.matmul(tf.transpose(hidden_state for word), self.primary_attention) + self.primary_bias, synonym )

		Then, calculate m for the word by summing this up, and then create h by concatenating the
		hidden state for the word and m
		"""
        h_hats = np.zeros((len(sentance), 300))
        out = self.primary_attention_dense_layer(hidden_states)
        for index, word in enumerate(sentance):
            synonym_embeddings = tf.nn.embedding_lookup(
                embedding_matrix, synonym_indices[word]
            )
            out_temp = tf.expand_dims(out[index, :], 1)
            coefficients = tf.math.exp(tf.matmul(synonym_embeddings, out_temp))
            m = tf.reduce_sum(coefficients * synonym_embeddings, 0)
            h_hat = m + hidden_states[index, :]
            h_hats[index, :] = h_hat
        return tf.convert_to_tensor(h_hats, dtype=np.float32)

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
        return tf.sum(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))

