import re

if __name__ == "__main__":
	with open('./data/imdb.txt', 'r') as myFile:
		data = myFile.read()

	data = re.split(r'\n', data)[:-1]

	print(len(data))

	with open('./data/train.txt', 'w') as myFile:
		for i in range(0, 25000):
			myFile.write(data[i]+'\n')

	with open('./data/test.txt', 'w') as myFile:
		for i in range(25000, 50000):
			myFile.write(data[i]+'\n')