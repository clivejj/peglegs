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
            tf.keras.layers.LSTM(units=self.h1, return_sequences=True), dtype=np.float32
        )
        self.primary_attention_layer = tf.keras.layers.Dense(self.h1)
        self.emotion_output_layer = tf.keras.layers.Dense(units=8, activation="sigmoid")
        self.sentiment_output_layer = tf.keras.layers.Dense(
            units=2, activation="sigmoid"
        )

    def call(self, batch_inputs, embedding_matrix, synonym_indices):

        for sentance in batch_inputs:
            # get embedding from word2vec embedding layer
            embeddings = tf.nn.embedding_lookup(embedding_matrix, sentance)
            hidden_states = tf.squeeze(self.biLSTM(tf.expand_dims(embeddings, 0)))
            a = self.primary_attention_layer(hidden_states)
            print(tf.shape(a))
            "CONTINUE HERE"
            return

            """for index, word in enumerate(sentance):
                synonym_embeddings = tf.nn.embedding_lookup(
                    embedding_matrix, synonym_indices[word]
                )
                print(hidden_states[:, index, :])
                print(self.primary_attention_layer(hidden_states[:, index, :]))
                print(synonym_embeddings)
                return
                coefficients = tf.tensordot(
                    self.primary_attention_layer(synonym_embeddings),
                    synonym_embeddings,
                    0,
                )
                print(tf.shape(coefficients))

                return

            return"""
        """
		TODO: Pass in hidden state output of biLSTM layer into primary attention layer for emotion
		and for sentiment primary attention layers

		Then, pass in the output of each of these, which will be h bar and h cap respectively,
		into their respective secondary attention layers to get H bar and H cap respective

		Add in dropout!
		"""

        h_bar = primary_attention((final_memory_output, final_carry_output))

        h_cap = primary_attention((final_memory_output, final_carry_output))

        H_bar = secondary_attention(h_bar)

        H_cap = secondary_attention(h_cap)

        emotion_logits = tf.nn.softmax(self.emotion_output_layer(H_bar))

        sentiment_logits = tf.nn.softmax(self.sentiment_output_layer(H_cap))

        return emotion_logits, sentiment_logits

    def primary_attention(self, hidden_state):
        """
		TODO: 
		for each of the 4 synonyms of a word (retrieved using wn.synset),
		calculate its attention using tf.math.exp( tf.matmul(
		tf.matmul(tf.transpose(hidden_state for word), self.primary_attention) + self.primary_bias, synonym )

		Then, calculate m for the word by summing this up, and then create h by concatenating the
		hidden state for the word and m
		"""
        pass

    def secondary_attention(self, primary_attention_output):
        """
		TODO:
		for each of the words in a sentence,
		calculate its attention using tf.math.exp(tf.math.tanh(
		tf.matmul(tf.transpose(hidden_state for word), self.secondary_attention) + self.secondary_bias))

		Then, calculate m for the word by summing this up, and then create h by concatenating the
		hidden state for the word and m
		"""
        pass

    def loss_function(self, labels, logits):
        # Calculate the sum of the loss by comparing the labels with the inputted logits
        return tf.sum(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))

