import tensorflow as tf
import numpy as np
from utils import construct_row


class Model(tf.keras.Model):
    def __init__(self, type):
        
        # Initialize the model, building off of the keras model
        
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

        self.primary_attention_dense_layer = tf.keras.layers.Dense(self.h1)

        self.secondary_attention_dense_layer = tf.keras.layers.Dense(
            1, activation = "tanh"
        )

        self.emotion_output_layer = tf.keras.layers.Dense(units=8)
        self.sentiment_output_layer = tf.keras.layers.Dense(units=1)
    # Run the forward training process. Depending on the type of the model,
    # the function will call the appropriate feed forward routine

    def call(self, sentence, embedding_matrix, synonym_indices):

        embeddings = tf.nn.embedding_lookup(embedding_matrix, sentence)
  
        hidden_states = tf.squeeze(self.biLSTM(tf.expand_dims(embeddings, 0)))
    
        # Make use of primary and secondary attention layers for both emotion and sentiment
        if(self.type == "full"):

            h_hats = self.primary_attention_emotion(
                sentence, hidden_states, embedding_matrix, synonym_indices
            )

            h_bars = self.primary_attention_emotion(
                sentence, hidden_states, embedding_matrix, synonym_indices
            )

            H_HAT = tf.expand_dims(self.secondary_attention_emotion(h_hats), 0)

            H_BAR = tf.expand_dims(self.secondary_attention_emotion(h_bars), 0)

            sentiment_logits = self.sentiment_output_layer(H_HAT)

            emotion_logits = self.emotion_output_layer(H_BAR)

            # print("Returning Full Call")
            return emotion_logits, sentiment_logits

        # Make use of only secondary attention layers for both emotion and sentiment
    
        elif(self.type == "multi_s"):
            
            H_HAT = tf.expand_dims(self.secondary_attention(hidden_states), 0)
        
            H_BAR = tf.expand_dims(self.secondary_attention(hidden_states), 0)

            sentiment_logits = self.sentiment_output_layer(H_HAT)

            emotion_logits = self.emotion_output_layer(H_BAR)
        
            # print("Returning Multi_S Call")
            return emotion_logits, sentiment_logits
       #
       #  Make use of both primary attention and secondary attention layers for only sentiment
       # 
        elif(self.type == "sentiment_only_p_and_s"):

            h_hats = self.primary_attention(
                sentence, hidden_states, embedding_matrix, synonym_indices
            )

            H_HAT = tf.expand_dims(self.secondary_attention(h_hats), 0)

            sentiment_logits = self.sentiment_output_layer(H_HAT)

            # print("Returning Sentiment_Only_P_and_S")
            return sentiment_logits

        #Make use of only the secondary attention layer for sentiment only 
        elif(self.type == "sentiment_only_s"):

            H_HAT = tf.expand_dims(self.secondary_attention(hidden_states), 0)

            sentiment_logits = self.sentiment_output_layer(H_HAT)

            # print("Returning Sentiment_Only_S")
            return sentiment_logits
        # 
        # Make use of Primary and secondary attention layers for only emotion
        # 
        elif(self.type == "emotion_only_p_and_s"):

            h_bars = self.primary_attention(
                sentence, hidden_states, embedding_matrix, synonym_indices
            )

            H_BAR = tf.expand_dims(self.secondary_attention(h_bars), 0)

            emotion_logits = self.emotion_output_layer(H_BAR)

            # print("Returning Emotion Only P and S")
            return emotion_logits
        # 
        # Make use of only secondary attention layer for emotion only
        # 
        elif(self.type == "emotion_only_s"):

            H_BAR = tf.expand_dims(self.secondary_attention(hidden_states), 0)

            emotion_logits = self.emotion_output_layer(H_BAR)

            # print("Returning Emotion Only S")
            return emotion_logits

        else:
            print("Invalid Model Type")
            return 0

    '''
    Apply primary attention layer rule to construct either h_bar or h_hat
    '''
    def primary_attention(
        self, sentence, hidden_states, embedding_matrix, synonym_indices
    ):
        out = self.primary_attention_dense_layer(hidden_states)

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
    '''
    Apply secondary attention layer rule to construct either H_BAR or H_Hat
    '''
    def secondary_attention(self, h_bars):
        coefficients = tf.math.exp(self.secondary_attention_dense_layer(h_bars))
        return tf.reduce_sum(coefficients * h_bars, 0)

    '''
    Calculate sigmoid loss
    '''
    def loss_function(self, labels, logits):
        # Calculate the sum of the loss by comparing the labels with the inputted logits
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        )

