# Importing dataset
import pandas as pd
messages=pd.read_csv('location', sep='\t', names=["label", "messages"])

# Preprocessing the dataset using regular expression
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
ps=PorterStemmer()
wordnet=WordNetLemmatizer()
###messages=nltk.sent_tokenize(messages)
corpus=[]
for i in range(len(messages)):
  review=re.sub('[^a-zA-Z]',' ',messages['messages'][i])
  review=review.lower()
  review=review.split()
  review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review= ' '.join(review)
  corpus.append(review)


# prompt: create a bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# prompt: test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# prompt: train using naive base classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)