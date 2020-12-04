from tensorflow.keras.layers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
import keras
from keras_self_attention import SeqSelfAttention
import numpy as np

class GAN_Models:
	def __init__(self, sequence_shape, latent_dim):
		self.sequence_shape = sequence_shape
		self.latent_dim = latent_dim

	def build_discriminator(self):
		model = Sequential()
		#model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,return_sequences=True)))
		model.add(LSTM(1024, input_shape=self.sequence_shape, return_sequences=True))
		model.add(Bidirectional(LSTM(512)))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.2))
		#model.add(Dense(1, activation='sigmoid'))
		#model.add(Attention())
		model.add(Dense(512))
		model.add(Dense(1, activation='sigmoid'))
		model.summary() 

		seq = Input(shape=self.sequence_shape)
		validity = model(seq)
		return Model(seq, validity)

	def build_generator(self):
		model = Sequential()
		#model.add(Bidirectional(LSTM(512)))
		model.add(Dense(512, input_dim=self.latent_dim))
		#model.add(LSTM(512, input_shape=self.sequence_shape, return_sequences=True))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(Dense(np.prod(self.sequence_shape), activation='tanh'))
		model.add(Reshape(self.sequence_shape))
		model.summary()
		noise = Input(shape=(self.latent_dim,))
		seq = model(noise)
		return Model(noise, seq)

