import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml #mnist datasetini yüklemek için
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784')
#1-2 dakika sürebilir.

print(mnist.data.shape)

# Parametre olarak dataframe ve ilgili veri fotoğrafının index numarasını alsın..
def showimage(dframe, index):    
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)

    plt.imshow(some_digit_image,cmap="binary")
    plt.axis("off")
    plt.show()

#örnek
img = showimage(mnist.data, 0)
print(img)

# test ve train oranı 1/7 ve 6/7
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)

print(type(train_img))

# Rakam tahminlerimizi check etmek için train_img dataframeini kopyalıyoruz, çünkü az sonra değişecek..
test_img_copy = test_img.copy()

show = showimage(test_img_copy,0)
print(show)


# Verilerimizi Scale etmemiz gerekiyor:

# Çünkü PCA scale edilmemiş verilerde hatalı sonuçlar verebiliyor bu nedenle mutlaka scaling işleminden geçiriyoruz. 
# Bu amaçla da StandardScaler kullanıyoruz...

scaler = StandardScaler()

# Scaler'ı sadece training set üzerinde fit yapmamız yeterli..
scaler.fit(train_img)

# Ama transform işlemini hem training sete hem de test sete yapmamız gerekiyor..
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


# PCA işlemini uyguluyoruz..
# Variance'ın 95% oranında korunmasını istediğimizi belirtiyoruz..

# Make an instance of the Model
pca = PCA(.95)

# PCA'i sadece training sete yapmamız yeterli: (1 dk sürebilir)
pca.fit(train_img)

# Bakalım 784 boyutu kaç boyuta düşürebilmiş (%95 variance'ı koruyarak tabiiki)
print(pca.n_components_)


# Şimdi transform işlemiyle hem train hem de test veri setimizin boyutlarını 784'ten 327'e düşürelim:
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)


# default solver çok yavaş çalıştığı için daha hızlı olan 'lbfgs' solverı seçerek logisticregression nesnemizi oluşturuyoruz.
logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter=10000)


# LogisticRegression Modelimizi train datamızı kullanarak eğitiyoruz:

# (Birkaç dk sürebilir)
logisticRegr.fit(train_img, train_lbl)


# #### Modelimiz eğitildi şimdi el yazısı rakamları makine öğrenmesi ile tanıma işlemini gerçekletirelim:

print(showimage(test_img_copy, 0))
logisticRegr.predict(test_img[0].reshape(1,-1))
#output#0 '0'

print(showimage(test_img_copy, 1))
logisticRegr.predict(test_img[1].reshape(1,-1))
#output#4 '4'

print(showimage(test_img_copy, 9900))
logisticRegr.predict(test_img[9900].reshape(1,-1))
#output#8 '8'

print(showimage(test_img_copy, 9999))
logisticRegr.predict(test_img[9999].reshape(1,-1))
#output#0 '0'


# Modelimizin doğruluk oranı (accuracy) ölçmek ..

# Modelimizin doğruluk oranı (accuracy) ölçmek için score metodunu kullanacağız:

result = logisticRegr.score(test_img, test_lbl)
print(result)
## %92
