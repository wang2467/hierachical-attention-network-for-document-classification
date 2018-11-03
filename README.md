# hierachical-attention-network-for-document-classification
Useful files:
    han.py -- HATT model
    train.py -- training script
    test.py -- testing script
    minibatch.py -- minibatch iterator
    preprocess.py -- build dictionary and encode data
    word2vec_gensim.py -- train word2vec model to initialize embedding matrix in HATT
    
Tested with IMDB dataset with 25000 training samples and testing samples. Got accuracy of approximately 43% on test set.
