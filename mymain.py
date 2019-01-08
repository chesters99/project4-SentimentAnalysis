import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# REQUIRED CODE
all_ = pd.read_csv('data.tsv', sep=' ', quotechar='"', escapechar='\\')
splits = pd.read_csv('splits.csv',sep='\t', dtype={'split_1':int,'split_2':int, 'split_3':int,})
s = 3

# Inititalise
start_time = time.time()
seed = 42

# Create train test from all and splits files
train = all_.loc[~all_.new_id.isin(splits['split_'+str(s)]), :].copy()
test  = all_.loc[ all_.new_id.isin(splits['split_'+str(s)]), all_.columns!='sentiment'].copy()
print('Split', s, 'Train shape:', train.shape, 'Test shape:', test.shape)

# read in vocab file
vocab_slim = np.loadtxt('myVocab.txt', delimiter=',', dtype=np.str)
print('Loaded vocab, size=', vocab_slim.shape[0])

# Create word vectors for test and train
train['review'] = train.review.str.replace('<br /><br />',' ')
test ['review'] =  test.review.str.replace('<br /><br />',' ')

stop_words= ['the','with','he','she','also','made','had','out','in','his','hers','there','was','then']

cv = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2), min_df=20, max_df=0.3)
X_train = cv.fit_transform(train.review)
X_test  = cv.transform(test.review)
vocab_all = np.array(cv.get_feature_names())
y_train = train.sentiment

# Remove all word columns that are not in vocabulary
keep_indices = np.where(np.in1d(vocab_all, vocab_slim))[0]
X_train = X_train[:, keep_indices]
X_test  = X_test [:, keep_indices]

# create model
model = LogisticRegression(penalty='l2',C=17, random_state=seed)
_ = model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:,1]

# save results to mysubmission file
df = pd.DataFrame({'new_id': test.new_id, 'prob': probs.round(5)})
df.to_csv('mysubmission.txt', index=False)
print('Created mysubmission.txt, rows=', df.shape[0],'in', round(time.time()-start_time,2),'secs') 
