import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer ##NLP için
from sklearn.ensemble import RandomForestClassifier ##assamble learning
from sklearn.metrics import roc_auc_score   ##yüzde hesaplama başarı
from bs4 import BeautifulSoup   
import re   ##(Regular Expressions)
import nltk 
from sklearn.model_selection import train_test_split ##train test
from nltk.corpus import stopwords   ##metin temizleme 



df = pd.read_csv("NLPlabeledData.tsv",delimiter="\t",quoting=3)
# result = df.head()
# print(result)

# result = len(df['review'])
# print(result)


##stopwords download
nltk.download("stopwords")

#Veri Temizleme İşlemleri
##Html taglarını kaldırmak için beautiful soup kullanıyoruz..

sample_review = df.review[0]
# print(review)


##HTML taglerini sildik..
sample_review = BeautifulSoup(sample_review).get_text()
# print(sample_review)


##Noktalama işaretini ve sayıları temizliyoruz. --regex kullanıyoruz...
sample_review = re.sub("[^a-zA-Z]", ' ', sample_review)
# print(sample_review)

##küçük harflere dönüştürüyoruz..
sample_review = sample_review.lower()
# print(sample_review)

##stopwords
sample_review = sample_review.split()
# print(sample_review)
# print(len(sample_review))

swords = set(stopwords.words("english"))
sample_review = [w for w in sample_review if w not in swords]
# print(sample_review)
# print(len(sample_review))


##Temizleme

def process(review):
    review = BeautifulSoup(review).get_text()
    review = re.sub("[^a-zA-Z]", ' ',review)
    review = review.lower()
    review = review.split()
    swords = set(stopwords.words("english"))
    review = [w for w in review if w not in swords]
    return(" ".join(review))

##trainig..
#her 1000 satır sonra işlem durumunu görüyoruz..

train_x_tum = []

for r in range(len(df["review"])):
    if (r+1) % 1000 == 0:
        print("No of reviews processed = ",r+1)
    train_x_tum.append(process(df["review"][r]))


##Train test

x = train_x_tum
y = np.array(df["sentiment"])
train_x ,test_x, y_train, y_test = train_test_split(x,y, test_size= 0.1 , random_state= 42)

## Bag of Words

vectorizer = CountVectorizer( max_features= 5000)

train_x = vectorizer.fit_transform(train_x)

# print(train_x)

##arraya dönüştür
train_x = train_x.toarray()
train_y = y_train
# print(train_x.shape , train_y.shape)
# print(train_y)

###Random Forest Modeli ..
#(1-2 dakika sürebilir...bekleyiniz..)
modelim = RandomForestClassifier(n_estimators= 100)
modelim.fit(train_x,train_y)

##test data

test_xx = vectorizer.transform(test_x)
# print(test_xx)

test_xx = test_xx.toarray()

## Prediction 

test_predict = modelim.predict(test_xx)
accuarry= roc_auc_score(y_test, test_predict)

print("Doğruluk oranı = %", accuarry * 100)
print("Tebrikler...")