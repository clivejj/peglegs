import tensorflow as tf
import numpy as np
from utils import construct_row


class Model(tf.keras.Model):
    def __init__(self, type):
        super(Model, self).__init__()

        # HyperParameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.embedding_size = 300
        self.h1 = 300
        self.batch_size = 64
        self.type = type
        # Trainable parameters

        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=int(self.h1 / 2), return_sequences=True,)
        )

        self.primary_attention_emotion_dense_layer = tf.keras.layers.Dense(self.h1)
        self.primary_attention_sentiment_dense_layer = tf.keras.layers.Dense(self.h1)

        self.secondary_attention_emotion_dense_layer = tf.keras.layers.Dense(
            1, activation="tanh"
        )
        self.secondary_attention_sentiment_dense_layer = tf.keras.layers.Dense(
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
        if(self.type == "full"):

            h_hats = self.primary_attention_sentiment(
                sentence, hidden_states, embedding_matrix, synonym_indices
            )

            h_bars = self.primary_attention_emotion(
                sentence, hidden_states, embedding_matrix, synonym_indices
            )

            H_HAT = tf.expand_dims(self.secondary_attention_sentiment(h_hats), 0)

            H_BAR = tf.expand_dims(self.secondary_attention_emotion(h_bars), 0)

            sentiment_logits = self.sentiment_output_layer(H_HAT)

            emotion_logits = self.emotion_output_layer(H_BAR)

            print("Returning Full Call")
            return emotion_logits, sentiment_logits


        elif(self.type == "multi_s"):
            
            H_HAT = tf.expand_dims(self.secondary_attention_sentiment(hidden_states), 0)
        
            H_BAR = tf.expand_dims(self.secondary_attention_emotion(hidden_states), 0)

            sentiment_logits = self.sentiment_output_layer(H_HAT)

            emotion_logits = self.emotion_output_layer(H_BAR)
        
            print("Returning Multi_S Call")
            return emotion_logits, sentiment_logits
       
        elif(self.type == "sentiment_only_p_and_s"):

            h_hats = self.primary_attention_sentiment(
                sentence, hidden_states, embedding_matrix, synonym_indices
            )

            H_HAT = tf.expand_dims(self.secondary_attention_sentiment(h_hats), 0)

            sentiment_logits = self.sentiment_output_layer(H_HAT)

            print("Returning Sentiment_Only_P_and_S")
            return sentiment_logits

        elif(self.type == "sentiment_only_s"):

            H_HAT = tf.expand_dims(self.secondary_attention_sentiment(hidden_states), 0)

            sentiment_logits = self.sentiment_output_layer(H_HAT)

            print("Returning Sentiment_Only_S")
            return sentiment_logits

        elif(self.type == "emotion_only_p_and_s"):

            h_bars = self.primary_attention_emotion(
                sentence, hidden_states, embedding_matrix, synonym_indices
            )

            H_BAR = tf.expand_dims(self.secondary_attention_emotion(h_bars), 0)

            emotion_logits = self.emotion_output_layer(H_BAR)

            print("Returning Emotion Only P and S")
            return emotion_logits

        elif(self.type == "emotion_only_s"):

            H_BAR = tf.expand_dims(self.secondary_attention_emotion(hidden_states), 0)

            emotion_logits = self.emotion_output_layer(H_BAR)

            print("Returning Emotion Only S")
            return emotion_logits

        else:
            print("Invalid Model Type")
            return








    def primary_attention_emotion(
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
        out = self.primary_attention_emotion_dense_layer(hidden_states)

        # construct first row of hs matrix
        h_bars = construct_row(
            0, sentence[0], out, hidden_states, embedding_matrix, synonym_indices
        )
        # construct remaining rows of hs matrix
        for index in range(1, len(sentence)):
            word = sentence[index]
            h_bar = construct_row(
                index, word, out, hidden_states, embedding_matrix, synonym_indices
            )
            h_bars = tf.concat([h_bars, h_bar], 0)

        return h_bars

    def primary_attention_sentiment(
        self, sentence, hidden_states, embedding_matrix, synonym_indices
    ):

        out = self.primary_attention_sentiment_dense_layer(hidden_states)

        h_hats = construct_row(
            0, sentence[0], out, hidden_states, embedding_matrix, synonym_indices
        )

        for index in range(1, len(sentence)):
            word = sentence[index]
            h_hat = construct_row(
                index, word, out, hidden_states, embedding_matrix, synonym_indices
            )
            h_hats = tf.concat([h_hats, h_hat], 0)

        return h_hats


    def secondary_attention_emotion(self, h_bars):
        """
		TODO:
		for each of the words in a sentence,
		calculate its attention using tf.math.exp(tf.math.tanh(
		tf.matmul(tf.transpose(hidden_state for word), self.secondary_attention) + self.secondary_bias))

		Then, calculate m for the word by summing this up, and then create h by concatenating the
		hidden state for the word and m
		"""
        coefficients = tf.math.exp(self.secondary_attention_emotion_dense_layer(h_bars))
        return tf.reduce_sum(coefficients * h_bars, 0)


    def secondary_attention_sentiment(self, h_hats):

        coefficients = tf.math.exp(self.secondary_attention_sentiment_dense_layer(h_hats))
        return tf.reduce_sum(coefficients * h_hats, 0)

    def loss_function(self, labels, logits):
        # Calculate the sum of the loss by comparing the labels with the inputted logits
        # print("Labels Type", labels)
        # print("Logits Type", logits)
        # return tf.convert_to_tensor(6.3)
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        )

