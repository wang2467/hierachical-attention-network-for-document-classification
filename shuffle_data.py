import numpy as np
import re

SHUFFLE_TIMES = 5

def main():
	with open('./data/train.txt', 'r') as myFile:
		data = myFile.read()

	data = re.split(r'\n', data)[:-1]

	for t in range(SHUFFLE_TIMES):
		result = [data[i] for i in np.random.permutation(len(data))]

	with open('./data/train_shuffled.txt', 'w') as myFile:
		for i in result:
			myFile.write(i + '\n')

	with open('./data/test.txt', 'r') as myFile:
		data = myFile.read()

	data = re.split(r'\n', data)[:-1]

	for t in range(SHUFFLE_TIMES):
		result = [data[i] for i in np.random.permutation(len(data))]

	with open('./data/test_shuffled.txt', 'w') as myFile:
		for i in result:
			myFile.write(i + '\n')


if __name__ == "__main__":
	main()
