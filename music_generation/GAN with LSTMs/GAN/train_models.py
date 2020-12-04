import numpy as np 
from itertools import chain
from keras.utils import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from .read_midi_convert_data import MIDI_Data
from .gan_models import GAN_Models

class Train:
	def __init__(self, rows):
		self.seq_length = rows
		self.seq_shape = (self.seq_length, 1)
		self.latent_dim = 1000
		self.optimizer = Adam(0.0002, 0.5)
		self.discriminator_losses = []
		self.generator_losses = []
		self.midi_data = MIDI_Data("data/anime/*.mid", "anime_notes.dat")
		self.gan_models = GAN_Models(self.seq_shape, self.latent_dim)
		self.discriminator = self.gan_models.build_discriminator()
		self.generator = self.gan_models.build_generator()
		self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
		self.discriminator.trainable = False
		input_placeholder = Input(shape=(self.latent_dim,))
		output_placeholder = self.generator(input_placeholder)
		validity = self.discriminator(output_placeholder)
		self.combined = Model(input_placeholder, validity)
		self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

	def prepare_music_sequences_for_traning(self):
		inp = []
		out = []
		seq_length = self.seq_length
		notes = self.midi_data.get_stored_midi_notes()
		n = len(set(notes))/2
		p_names = set()
		for note in notes:
			p_names.add(note)
		p_names = sorted(p_names)
		note_dict = {}
		i = 0
		for pitch in p_names:
			note_dict[pitch]=i
			i+=1
		end = len(notes) - seq_length
		for i in range(0, end):
			s_in = notes[i:i + seq_length]
			s_out = notes[i + seq_length]
			li = []
			for ch in s_in:
				li.append(note_dict[ch])
			inp.append(li)
			out.append(note_dict[s_out])
		size = (len(inp), seq_length, 1)
		inp = (np.reshape(inp, size) - n)/n
		#inp is the normalised input
		out = to_categorical(out)
		tup = (inp, out)
		return tup

	def train_models(self, epochs, batch_size=128):

		X_train, y_train = self.prepare_music_sequences_for_traning()
		real = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))
		for epoch in range(epochs):
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			real_seqs = X_train[idx]
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
			gen_seqs = self.generator.predict(noise)
			#Train discriminator
			d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
			d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
			# Train generator 
			g_loss = self.combined.train_on_batch(noise, real)
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
			self.discriminator_losses.append(d_loss[0])
			self.generator_losses.append(g_loss)

		self.generate_midi_file_with_trained_network()
		self.plot_losses_of_models()

	def generate_midi_file_with_trained_network(self):
		notes = self.midi_data.get_stored_midi_notes()
		n = len(set(notes))
		p_names = set()
		for note in notes:
			p_names.add(note)
		p_names = sorted(p_names)
		note_dict = {}
		i = 0
		for pitch in p_names:
			note_dict[i]=pitch
			i+=1
		noise = np.random.normal(0, 1, (1, self.latent_dim))
		pred = self.generator.predict(noise)[0]
		p_n = []
		for p in pred:
			p_n.append(((p + 1)*n)/2)
		final_p = []
		for n in p_n:
			final_p.append(note_dict[int(n)])
		
		self.midi_data.create_midi_file(final_p, 'generated_file')

	def plot_losses_of_models(self):
		plt.title("GAN Loss per Epoch")
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.plot(self.discriminator_losses, c='red')
		plt.plot(self.generator_losses, c='blue')
		plt.legend(['Discriminator', 'Generator'])
		plt.savefig('GAN_models_Loss_per_Epoch_final.png', transparent=True)
		plt.close()


	


