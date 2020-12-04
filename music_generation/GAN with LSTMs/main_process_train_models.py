from GAN.train_models import Train

if __name__ == "__main__":
	train = Train(100)
	train.train_models(epochs=5)