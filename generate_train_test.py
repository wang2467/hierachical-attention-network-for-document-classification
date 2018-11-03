import os

if __name__ == "__main__":
	data = []
	labels = []
	dir_paths = ['./data/train/pos/', './data/train/neg/', './data/test/pos/', './data/test/neg/']

	for dir_path in dir_paths:
		for i in os.listdir(dir_path):
			with open(dir_path + str(i), 'r') as myFile:
				data.append(myFile.read())
			label = i.strip().split('_')[1].split('.')[0]
			labels.append(str(label))

	print(len(data))
	print(len(labels))

	with open('./imdb_data/train_test.txt', 'w') as myFile:
		for idx, sent in enumerate(data):
			myFile.write(sent+'---'+labels[idx]+'\n')