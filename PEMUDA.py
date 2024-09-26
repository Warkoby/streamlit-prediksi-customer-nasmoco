#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df = pd.read_excel("datacustomerPemuda.xlsx")
df.head()


# In[3]:


df.describe()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


#null values
df.isnull().values.any()


# In[9]:


# Menghitung jumlah masing-masing jenis kelamin
gender_count = df['JenisKelamin'].value_counts()


# In[10]:


# Membuat countplot untuk Jenis Kelamin
plt.figure(figsize=(8,6))
sns.countplot(x='JenisKelamin', data=df, palette='pastel')

# Menambahkan judul dan label
plt.title('Perbandingan Jumlah Jenis Kelamin', fontsize=16)
plt.xlabel('Jenis Kelamin', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)

# Menampilkan plot
plt.show()


# In[11]:


# Membuat countplot untuk JenisMobil dengan rotasi label x-axis
plt.figure(figsize=(10,6))
sns.countplot(x='JenisMobil', data=df, palette='pastel')

# Menambahkan judul dan label
plt.title('Perbandingan Jumlah Berdasarkan Jenis Mobil', fontsize=16)
plt.xlabel('Jenis Mobil', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)

# Memutar label sumbu x sebanyak 90 derajat
plt.xticks(rotation=90)
plt.show()


# In[12]:


# Filter data untuk TahunBeli dari 2019 hingga 2023
df_filtered = df[(df['TahunBeli'] >= 2019) & (df['TahunBeli'] <= 2023)]


# In[13]:


# Membuat countplot untuk TahunBeli
plt.figure(figsize=(8,6))
sns.countplot(x='TahunBeli', data=df_filtered, palette='pastel')

# Menambahkan judul dan label
plt.title('Jumlah Service Mobil Berdasarkan 5 Tahun Terakhir (2019 - 2023)', fontsize=16)
plt.xlabel('Tahun Service', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)

# Menampilkan plot
plt.show()


# In[14]:


# Mengelompokkan umur dalam rentang 5 tahun
bins = range(0, df['Umur'].max() + 5, 5)  # Rentang 0 hingga umur maksimal, dengan interval 5
labels = [f'{i}-{i+4}' for i in bins[:-1]]  # Label rentang umur


# In[15]:


# Membuat kolom baru untuk kelompok umur
df_filtered['KelompokUmur'] = pd.cut(df_filtered['Umur'], bins=bins, labels=labels, right=False)


# In[16]:


# Membuat count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df_filtered, x='KelompokUmur', order=labels, palette='pastel')

# Menambahkan judul dan label
plt.title('Distribusi Umur Berdasarkan Kelompok (65+ hingga Maksimum)', fontsize=16)
plt.xlabel('Kelompok Umur')
plt.ylabel('Jumlah')

# Memutar label sumbu x agar lebih mudah dibaca
plt.xticks(rotation=45)

# Menampilkan count plot
plt.tight_layout()
plt.show()


# In[17]:


# Create the countplot for the 'Pekerjaan' column
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Hobi', order=df['Hobi'].value_counts().index, palette='bright')

# Add title and labels
plt.title('Hobi Pelanggann', fontsize=16)
plt.xlabel('Hobi', fontsize=14)
plt.ylabel('Jumlah', fontsize=14)

# Rotate xb-axis labels if needed
plt.xticks(rotation=90, ha='right')

# Show plot
plt.show()


# In[18]:


# Create the countplot for the 'Pekerjaan' column
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Olahraga', order=df['Olahraga'].value_counts().index, palette='bright')

# Add title and labels
plt.title('Olahraga yang digemari customer', fontsize=16)
plt.xlabel('Olahraga', fontsize=14)
plt.ylabel('Jumlah', fontsize=14)

# Rotate xb-axis labels if needed
plt.xticks(rotation=90, ha='right')

# Show plot
plt.show()


# In[19]:


# Menghitung distribusi nilai unik pada kolom 'Warna'
warna_counts = df['Warna'].value_counts()

# Menghitung total data untuk mendapatkan persentase
total = warna_counts.sum()

# Pisahkan warna yang frekuensinya lebih besar atau sama dengan 2%
warna_counts_filtered = warna_counts[warna_counts / total >= 0.02]

# Gabungkan warna yang frekuensinya kurang dari 2% ke dalam satu kategori 'Warna Lainnya'
warna_lainnya = warna_counts[warna_counts / total < 0.02].sum()

# Tambahkan kategori 'Warna Lainnya' jika ada data yang kurang dari 2%
if warna_lainnya > 0:
    warna_counts_filtered['Warna Lainnya'] = warna_lainnya


# In[20]:



# Membuat pie chart
plt.figure(figsize=(8, 8))
plt.pie(warna_counts_filtered, labels=warna_counts_filtered.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

# Menambahkan judul
plt.title('Distribusi Warna Mobil', fontsize=16)

# Menampilkan pie chart
plt.axis('equal')  # Agar pie chart berbentuk lingkaran sempurna
plt.show()


# In[21]:


# Hitung jumlah setiap pekerjaan
pekerjaan_counts = df_filtered['Pekerjaan'].value_counts()


# In[22]:


# Combine 'Pekerjaan' with less than 0.5% of total into 'Pekerjaan Lainnya'
total_purchases = df_filtered['Pekerjaan'].count()
threshold = 0.005 * total_purchases  # 0.5% threshold


# In[23]:


# Identify pekerjaan with count below threshold
pekerjaan_counts_adjusted = pekerjaan_counts.copy()
pekerjaan_counts_adjusted[pekerjaan_counts < threshold] = 0  # Set small values to zero temporarily
pekerjaan_lainnya = pekerjaan_counts[pekerjaan_counts < threshold].sum()


# In[24]:


# Replace low frequency jobs with 'Pekerjaan Lainnya'
pekerjaan_counts_adjusted = pekerjaan_counts_adjusted[pekerjaan_counts_adjusted >= threshold]
pekerjaan_counts_adjusted['Pekerjaan Lainnya'] = pekerjaan_lainnya


# In[25]:


# Scatter plot of adjusted pekerjaan
plt.figure(figsize=(10, 6))
plt.scatter(pekerjaan_counts_adjusted.index, pekerjaan_counts_adjusted.values, color='blue', s=100)

# Add labels and title
plt.xlabel('Pekerjaan', fontsize=12)
plt.ylabel('Jumlah Pembelian', fontsize=12)
plt.title('Scatter Plot: Pekerjaan (Termasuk Pekerjaan Lainnya) yang Membeli Mobil dalam 5 Tahun Terakhir', fontsize=16)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the scatter plot
plt.tight_layout()
plt.show()


# In[26]:


# Menghitung matriks korelasi
correlation_matrix = df.corr()
correlation_matrix


# In[27]:


# Membuat heatmap dari matriks korelasi
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Menambahkan judul
plt.title('HeatMap', fontsize=16)

# Menampilkan heatmap
plt.show()


# In[28]:


data=df.drop(['DealerPembelian', 'JenisKelamin', 'Hobi', 'Pekerjaan', 'Olahraga',
       'Umur', 'JenisMobil', 'Warna','Diskon', 'Asuransi',
       'TahunBeli', 'ServiceBP_LIGHT', 'ServiceBP_MEDIUM', 'ServiceBP_HEAVY',
       'ServiceGR', 'Service10K', 'Service20K', 'Service30K', 'Service40K',
       'Service50K', 'Service60K', 'Service70K', 'Service80K', 'Service90K',
       'Service100K', 'Service100Klebih'], axis=1)


# In[29]:


data.head()


# In[30]:


data1=data[['Pembayaran', 'BeliUlang']]


# In[31]:


X = data1.drop(columns=['BeliUlang'])
y = df['BeliUlang']


# In[32]:


X_encoded = pd.get_dummies(X)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[34]:


# Standardize the data (for models that benefit from scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[35]:


# Membuat model Linear Regression
lr_model = LinearRegression()


# In[36]:


# Melatih model pada data train
lr_model.fit(X_train, y_train)

# Membuat prediksi
y_pred_lr = lr_model.predict(X_test)

# Membulatkan hasil prediksi ke nilai 0 dan 1 karena Linear Regression biasanya untuk regresi
y_pred_lr_rounded = [1 if pred >= 0.5 else 0 for pred in y_pred_lr]

# Menghitung akurasi
accuracy_lr = accuracy_score(y_test, y_pred_lr_rounded)
print(f'Accuracy of Linear Regression: {accuracy_lr * 100:.2f}%')


# In[37]:


# Membuat model Decision Tree
dt_model = DecisionTreeClassifier()


# In[38]:


# Melatih model pada data train
dt_model.fit(X_train, y_train)

# Membuat prediksi
y_pred_dt = dt_model.predict(X_test)

# Menghitung akurasi
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Accuracy of Decision Tree: {accuracy_dt * 100:.2f}%')


# In[39]:


# Membuat model Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[40]:


#### Melatih model pada data train
rf_model.fit(X_train, y_train)

# Membuat prediksi
y_pred_rf = rf_model.predict(X_test)

# Menghitung akurasi
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Menghitung akurasi
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Accuracy of Random Forest: {accuracy_dt * 100:.2f}%')


# In[ ]:




