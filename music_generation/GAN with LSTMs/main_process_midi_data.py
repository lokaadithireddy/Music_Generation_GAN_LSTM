from GAN.read_midi_convert_data import MIDI_Data

if __name__ == "__main__":
	midi_data = MIDI_Data("data/anime/*.mid", "anime_notes.dat")
	midi_data.store_all_notes_to_data_file()
