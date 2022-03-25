# Made some modifications from SSR-net
# source file (https://github.com/shamangary/SSR-Net/blob/master/training_and_testing/TYY_callbacks.py)
# source file LICENSE ( https://github.com/shamangary/SSR-Net/blob/master/LICENSE)
import tensorflow.keras as tf_keras
from tensorflow.keras.backend import get_value, set_value

class DecayLearningRate(tf_keras.callbacks.Callback):
	def __init__(self, startEpoch):
		self.startEpoch = startEpoch

	def on_train_begin(self, logs={}):
		return
	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		
		if epoch in self.startEpoch:
			if epoch == 0:
				ratio = 1
			else:
				ratio = 0.1
			LR = get_value(self.model.optimizer.lr)
			set_value(self.model.optimizer.lr,LR*ratio)
		return
	def on_epoch_end(self, epoch, logs={}):
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
