import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes

df = pd.read_csv("segmentation_data.csv")
# print(df.head())


result = df.isnull().sum()
# print(result)

#  Income ve Age Data Normalization

temp = df[['ID','Age','Income']]
# print(temp)

normalization = MinMaxScaler()

normalization.fit(df[['Age']])
df['Age'] = normalization.transform(df[['Age']])

normalization.fit(df[['Income']])
df['Income'] = normalization.transform(df[['Income']])



# Drop ID before analysis..

df = df.drop(['ID'], axis=1)
# print(df)

mark_array= df.values

mark_array[:, 2] = mark_array[:, 2].astype(float)
mark_array[:, 4] = mark_array[:, 4].astype(float)

# Build our model..

kproto = KPrototypes(n_clusters=15,verbose=2,max_iter=30)
clusters = kproto.fit_predict(mark_array, categorical=[0,1,3,5,6])

# print(kproto.cluster_centroids_)
# print(len(kproto.cluster_centroids_))

c_dict = []

for c in clusters:
    c_dict.append(c)

df['cluster']=c_dict


# Put original columns from temp to df:
df[['ID','Age','Income']]= temp
result= df[df['cluster'] == 0].head(10)
# print(result)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
df5 = df[df.cluster==4]
df6 = df[df.cluster==6]
df7 = df[df.cluster==7]
df8 = df[df.cluster==8]
df9 = df[df.cluster==9]
df10 = df[df.cluster==10]
df11 = df[df.cluster==11]
df12 = df[df.cluster==12]
df13 = df[df.cluster==13]
df14 = df[df.cluster==14]
df15 = df[df.cluster==15]

plt.figure(figsize=(12,12))
plt.xlabel('Age')
plt.ylabel('Income')

plt.scatter(df1.Age, df1['Income'],color='green', alpha = 0.4)
plt.scatter(df2.Age, df2['Income'],color='red', alpha = 0.4)
plt.scatter(df3.Age, df3['Income'],color='gray', alpha = 0.4)
plt.scatter(df4.Age, df4['Income'],color='orange', alpha = 0.4)
plt.scatter(df5.Age, df5['Income'],color='yellow', alpha = 0.4)
plt.scatter(df6.Age, df6['Income'],color='cyan', alpha = 0.4)
plt.scatter(df7.Age, df7['Income'],color='magenta', alpha = 0.4)
plt.scatter(df8.Age, df8['Income'],color='gray', alpha = 0.4)
plt.scatter(df9.Age, df9['Income'],color='purple', alpha = 0.4)
plt.scatter(df10.Age, df10['Income'],color='brown', alpha = 0.4)
plt.scatter(df11.Age, df11['Income'],color='black', alpha = 0.4)
plt.scatter(df12.Age, df12['Income'],color='lightpink', alpha = 0.4)
plt.scatter(df13.Age, df13['Income'],color='indigo', alpha = 0.4)
plt.scatter(df14.Age, df14['Income'],color='lightcoral', alpha = 0.4)
plt.scatter(df15.Age, df15['Income'],color='peru', alpha = 0.4)
plt.legend()
plt.show()