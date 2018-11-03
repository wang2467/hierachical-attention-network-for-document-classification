from preprocess import Preprocessor
import gensim

def main():
	p = Preprocessor('./data/train.txt', 'build')
	p._clean_data()

	sents = [sent for doc in p.cleaned_data for sent in doc]

	model = gensim.models.Word2Vec(sents, size=200, min_count=6, window=10, workers=10)
	model.train(sents, total_examples=len(sents), epochs=10)

	model.save('./data/w2v.model')

	word_vectors = gensim.models.Word2Vec.load('./data/w2v.model')

	print(word_vectors['the'])
	print(list(model.wv.vocab))

	model = gensim.models.Word2Vec.load('./data/w2v.model')
	print(model['the'])
	print(len(model.wv.vocab))

if __name__ == "__main__":
	main()
