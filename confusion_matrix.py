from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def main():
	with open('./data/result.txt', 'r') as myFile:
		result = myFile.read().splitlines()

	y_pred, y_label = [], []

	for res in result:
		y_pred.append(res.split('---')[0])
		y_label.append(res.split('---')[1])

	cm = confusion_matrix(y_label, y_pred, labels=[str(i) for i in range(0, 4)].extend([str(i) for i in range(6, 10)]))

	plt.figure()
	plt.imshow(cm, cmap=plt.get_cmap('Blues'))
	plt.title('confusion matrix without normalization')
	plt.colorbar()
	tick_marks = np.arange(8)
	plt.xticks(tick_marks, ['1', '2', '3', '4', '7', '8', '9', '10'], rotation=45)
	plt.yticks(tick_marks, ['1', '2', '3', '4', '7', '8', '9', '10'])

	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			plt.text(j, i, '{}'.format(cm[i, j]), horizontalalignment="center", color='white' if cm[i, j] > cm.max() / 2 else 'black')


	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

if __name__ == "__main__":
	main()