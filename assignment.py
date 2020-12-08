import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from utils import unpickle, get_batch, setup
from preprocessing import get_vec
from model import Model
from sklearn.metrics import f1_score, recall_score, precision_score


def train(
	model, train_inputs, emotion_labels, sentiment_labels, embeddings, synonym_indices
):

	for batch_num in range(int(len(train_inputs) / model.batch_size)):

		print("Batch Number", batch_num)
		batch_inputs, batch_emotion_labels, batch_sentiment_labels = get_batch(
			train_inputs, emotion_labels, sentiment_labels, batch_num, model.batch_size
		)

		if model.type == "full" or model.type == "multi_s":
			with tf.GradientTape() as tape:
				emotion_batch_loss = 0
				sentiment_batch_loss = 0
				acc = []

				for index, tweet in enumerate(batch_inputs):
					emotion_logits, sentiment_logit = model.call(
						tweet, embeddings, synonym_indices
					)
					emotion_batch_loss += model.loss_function(
						tf.expand_dims(batch_emotion_labels[index], 0), emotion_logits
					)
					sentiment_batch_loss += model.loss_function(
						tf.expand_dims(batch_sentiment_labels[index], 0), sentiment_logit
					)
					acc += [sentiment_logit]
				batch_loss = (emotion_batch_loss/ 4) + sentiment_batch_loss

			acc = np.squeeze(
				tf.cast(tf.math.sigmoid(tf.convert_to_tensor(acc)) > 0.5, np.float32), 2
			)
			accuracy = tf.reduce_sum(tf.cast(acc == batch_sentiment_labels, tf.int32))
			print("Batch Accuracy", accuracy / 64)
			print("Batch Loss", batch_loss)
			gradients = tape.gradient(batch_loss, model.trainable_variables)
			model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		elif model.type == "sentiment_only_p_and_s" or model.type == "sentiment_only_s":
			with tf.GradientTape() as tape:
				sentiment_batch_loss = 0
				acc = []

				for index, tweet in enumerate(batch_inputs):
					sentiment_logit = model.call(
						tweet, embeddings, synonym_indices
					)
					sentiment_batch_loss += model.loss_function(
						tf.expand_dims(batch_sentiment_labels[index], 0), sentiment_logit
					)
					acc += [sentiment_logit]
				batch_loss = sentiment_batch_loss
			acc = np.squeeze(
				tf.cast(tf.math.sigmoid(tf.convert_to_tensor(acc)) > 0.5, np.float32), 2
			)  
			accuracy = tf.reduce_sum(tf.cast(acc == batch_sentiment_labels, tf.int32))
			print("Batch Accuracy", accuracy / 64)
			print("Batch Loss", batch_loss)
			gradients = tape.gradient(batch_loss, model.trainable_variables)
			model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		elif model.type == "emotion_only_p_and_s" or model.type == "emotion_only_s" :
			with tf.GradientTape() as tape:
				emotion_batch_loss = 0
				acc = []

				for index, tweet in enumerate(batch_inputs):
					emotion_logits = model.call(
						tweet, embeddings, synonym_indices
					)
					emotion_batch_loss += model.loss_function(
						tf.expand_dims(batch_emotion_labels[index], 0), emotion_logits
					)
				batch_loss = emotion_batch_loss / 4
			# acc = np.squeeze(
			# 	tf.cast(tf.math.sigmoid(tf.convert_to_tensor(acc)) > 0.5, np.float32), 2
			# )  
			print("Batch Loss", batch_loss)
			gradients = tape.gradient(batch_loss, model.trainable_variables)
			model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		else: 
			print("Invalid Model Type")
			return



def test(
	model, test_inputs, emotion_labels, sentiment_labels, embeddings, synonym_indices
):
	emotionF1 = 0
	emotionRecall = 0
	sentimentAcc = 0

	for index, tweet in enumerate(test_inputs):
		print("Tweet", index)
		if model.type == "full" or model.type == "multi_s":
			emotion_logits, sentiment_logits = model.call(
				tweet, embeddings, synonym_indices
			)

			emotion_predictions = tf.squeeze(
				tf.cast(tf.math.sigmoid(emotion_logits) > 0.5, tf.float32), 0
			)

			emotion_f1_batch = f1_score(emotion_labels[index, :], emotion_predictions)
			emotion_recall_batch = recall_score(emotion_labels[index, :], emotion_predictions)
			emotionF1 += emotion_f1_batch
			emotionRecall += emotion_recall_batch

			sentiment_predictions = tf.cast(
				tf.math.sigmoid(sentiment_logits) > 0.5, tf.float32
			)
			sentiment_acc_batch = tf.cast(
				sentiment_labels[index, :] == sentiment_predictions, tf.float32
			)
			sentimentAcc += sentiment_acc_batch

		elif model.type == "sentiment_only_p_and_s" or model.type == "sentiment_only_s":
			sentiment_logits = model.call(
				tweet, embeddings, synonym_indices
			)
			sentiment_predictions = tf.cast(
				tf.math.sigmoid(sentiment_logits) > 0.5, tf.float32
			)
			sentiment_acc_batch = tf.cast(
				sentiment_labels[index, :] == sentiment_predictions, tf.float32
			)
			sentimentAcc += sentiment_acc_batch


		elif model.type == "emotion_only_p_and_s" or model.type == "emotion_only_s":
			emotion_logits = model.call(
				tweet, embeddings, synonym_indices
			)
			emotion_predictions = tf.squeeze(
				tf.cast(tf.math.sigmoid(emotion_logits) > 0.5, tf.float32), 0
			)

			emotion_f1_batch = f1_score(emotion_labels[index, :], emotion_predictions)
			emotion_recall_batch = recall_score(emotion_labels[index, :], emotion_predictions)
			emotionF1 += emotion_f1_batch
			emotionRecall += emotion_recall_batch
		else:
			print("Invalid Model Type")
			return

		# if index % 100 == 0:
		#     print(sentimentAcc / index)
	if model.type == "full" or model.type == "multi_s":
		average_emotion_F1 = emotionF1 / len(test_inputs)
		print("Average Emotion F1", average_emotion_F1)
		average_emotion_Recall = emotionRecall / len(test_inputs)
		print("Average Emotion Recall", average_emotion_Recall)
		average_sentiment_acc = sentimentAcc / len(test_inputs)
		print("Average Sentiment Acc", average_sentiment_acc)

	elif model.type == "sentiment_only_p_and_s" or model.type == "sentiment_only_s":
		average_sentiment_acc = sentimentAcc / len(test_inputs)
		print("Average Sentiment Acc", average_sentiment_acc)

	elif model.type == "emotion_only_p_and_s" or model.type == "emotion_only_s":
		average_emotion_F1 = emotionF1 / len(test_inputs)
		print("Average Emotion F1", average_emotion_F1)
		average_emotion_Recall = emotionRecall / len(test_inputs)
		print("Average Emotion Recall", average_emotion_Recall)

	else: 
		print("Invalid Model Type")
		return
  


def main():
	# Unpickles data from file, stores it as dictionary of length 4
	setup(isTraining=True, overwrite=False)
	print("done processing training data")
	data = unpickle("data/training.pickle")
	# A dictionary that keys every word in our vocabulary to an index
	train_vocab = data[0]
	# A list of the tweets that we will be training on (2914 tweets)
	train_sentences = data[1]
	"""for i in range(len(train_sentences)):
		train_sentences[i] = tf.convert_to_tensor(train_sentences[i], tf.int32)"""
	# print("Sentences", len(sentences))
	# An embedding matrix that maps each word to a 300 Dimensional Embedding
	train_embeddings = tf.convert_to_tensor(data[2], tf.float32)
	# A dictionary that maps the index of a word to a list containing the indices of its 4 synonyms
	train_synonym_indices = tf.convert_to_tensor(data[3], tf.int32)

	# A list of sentiment labels corresponding to tweets; labels can be -1 (negative), 0 (objective), or (1) positive
	# (2914, 1)
	train_sentiment_labels = tf.convert_to_tensor(data[4], tf.float32)
	# A list of emotion labels corresponding to tweets; each label has 8 slots, where a 1 in that position corresponds to that
	# emotion being labelled. So, each tweet can be associated to several different emotions
	# Shape (2914, 8)
	train_emotion_labels = tf.convert_to_tensor(data[5], tf.float32)
	data = None

	model = Model("emotion_only_s")

	train(
		model,
		train_sentences,
		train_emotion_labels,
		train_sentiment_labels,
		train_embeddings,
		train_synonym_indices,
	)

	setup(isTraining=False, overwrite=False)
	print("done processing testing data")
	data = unpickle("data/testing.pickle")

	test_vocab = data[0]
	test_sentences = data[1]
	"""for i in range(len(train_sentences)):
		train_sentences[i] = tf.convert_to_tensor(train_sentences[i], tf.int32)"""
	test_embeddings = tf.convert_to_tensor(data[2], tf.float32)
	test_synonym_indices = tf.convert_to_tensor(data[3], tf.int32)
	test_sentiment_labels = tf.convert_to_tensor(data[4], tf.float32)
	test_emotion_labels = tf.convert_to_tensor(data[5], tf.float32)
	data = None

	test(
		model,
		test_sentences,
		test_emotion_labels,
		test_sentiment_labels,
		test_embeddings,
		test_synonym_indices,
	)


if __name__ == "__main__":
	main()
