import pandas as pd

#loading data: data is tab separated
df=pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])


#data cleaning---
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stema=PorterStemmer()
word_list=[]
for i in range(0,len(df)):
    words=re.sub('[^a-zA-Z]',' ',df['message'][i])
    words=words.lower()
    words=words.split()

    words=[stema.stem(word) for word in words if word not in stopwords.words('english')]
    words=' '.join(words)
    word_list.append(words)


#bag of words---gives rows as sentences and columns as words
from sklearn.feature_extraction.text import CountVectorizer
obj=CountVectorizer(max_features=5000)  #top 5000 frequent words
x=obj.fit_transform(word_list).toarray()   #5572 sentences and 5000 words/columns


#output data
y=pd.get_dummies(df['label'])
y=y.iloc[:,1].values   #ham is 0 and spam is 1



#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)


#training model using naive bayes classifier
from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)


#confusion matrix
from sklearn.metrics import confusion_matrix
a=confusion_matrix(y_test, y_pred)


#evaluation
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)






