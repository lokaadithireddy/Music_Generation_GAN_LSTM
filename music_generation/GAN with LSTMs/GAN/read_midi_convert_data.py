import glob
#from music21 import converter, instrument, note, chord, stream
from music21 import *
import pickle


class MIDI_Data:

	def __init__(self, file_patterns, file_storage_location):
		self.file_patterns = file_patterns
		self.file_storage_location = file_storage_location

	def read_midi_data_to_file(self):
		notes = []
		for file in glob.glob(self.file_patterns):
			print(file)
			midi = converter.parse(file)
			parts = instrument.partitionByInstrument(midi)
			print(midi, parts)
			notes_in_score = None
			parts = instrument.partitionByInstrument(midi)
			if parts:
				notes_in_score = parts.parts[0].recurse()
			else:
				notes_in_score = midi.flat.notes
			print (notes_in_score)
			for element in notes_in_score:
				if isinstance(element, note.Note):
					notes.append(str(element.pitch))
				elif isinstance(element, chord.Chord):
					notes.append('.'.join(str(n) for n in element.normalOrder))
		print(notes)
		return notes

	def store_all_notes_to_data_file(self):
		with open(self.file_storage_location, "wb") as fp:
			pickle.dump(self.read_midi_data_to_file(), fp)

	def get_stored_midi_notes(self):
		with open(self.file_storage_location, "rb") as fp:
			return pickle.load(fp)

	def create_midi_file(self, output, f_name):
		o = 0
		result = []
		for val in output:
			b_val = ('.' in val[0]) or val[0].isdigit()
			if b_val:
				c_notes = []
				for v in val[0].split('.'):
					v = int(v)
					get_note = note.Note(v)
					get_note.storedInstrument = instrument.Piano()
					c_notes.append(get_note)
				get_chord = chord.Chord(c_notes)
				get_chord.offset = o
				result.append(get_chord)
			else:
				get_note = note.Note(val[0])
				get_note.offset = o
				get_note.storedInstrument = instrument.Piano()
				result.append(get_note)
			o += 0.5
		midi_stream = stream.Stream(result)
		midi_stream.write('midi', fp='{}.mid'.format(f_name))


