import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

en_data = pd.read_csv('en_trigram_count.tsv', delimiter='\t', header=None, index_col=0)
es_data = pd.read_csv('es_trigram_count.tsv', delimiter='\t', header=None, index_col=0)
fr_data = pd.read_csv('fr_trigram_count.tsv', delimiter='\t', header=None, index_col=0)
pt_data = pd.read_csv('pt_trigram_count.tsv', delimiter='\t', header=None, index_col=0)

X_train = np.array([en_data[2], es_data[2], fr_data[2], pt_data[2]])
y_train = np.array(['en', 'es', 'fr', 'pt'])

nb = MultinomialNB(alpha=1.0, fit_prior=False, class_prior=None)

nb.fit(X_train, y_train)
print('Verification: ', nb.predict(X_train), y_train)

sentences = [
    'El cine esta abierto.',
    'Tu vais à escola hoje.',
    'Tu vais à escola hoje pois já estás melhor.',
    'English is easy to learn.',
    'Tu vas au cinéma demain matin.',
    'É fácil de entender.',
]

vectorizer = CountVectorizer(ngram_range=(3,3), analyzer='char', vocabulary=en_data[1])

X_test = vectorizer.fit_transform(sentences)
y_test = np.array(['es', 'pt', 'pt', 'en', 'fr', 'pt'])
#print(vectorizer.get_feature_names())
#print(X_test.toarray())

print('Predictions: ', nb.predict(X_test), 'Real: ', y_test)
probs_sorted = np.sort(nb.predict_proba(X_test))
#print(probs_sorted)
class_margin = [probs_sorted[0,-1]-probs_sorted[0,-2],\
                probs_sorted[1,-1]-probs_sorted[1,-2],\
                probs_sorted[2,-1]-probs_sorted[2,-2],\
                probs_sorted[3,-1]-probs_sorted[3,-2],\
                probs_sorted[4,-1]-probs_sorted[4,-2],\
                probs_sorted[5,-1]-probs_sorted[5,-2]]

print('Classification margin: ', class_margin)