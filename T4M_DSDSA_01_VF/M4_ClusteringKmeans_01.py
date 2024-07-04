# Model Clustering: K-Means untuk Mengelompokkan Tamu Berdasarkan Pola Pemesanan

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Upload data: Memuat dataset hotel_bookings.csv.
data = pd.read_csv('D:\\Kuliah\\Data Science pada Domain Spesifik\\T4\\DSDS (Steffi)\\hotel_bookings.csv')

# 2. Observasi data: Melihat isi data dan melakukan analisis deskriptif.
print(data.info())
print(data.describe())
print(data.isnull().sum())

# 3. Preprocessing data: Menangani nilai yang hilang dan outliers.
data = data[(data['lead_time'] <= 365)]
data = data[(data['adr'] <= 5000)]
data['children'] = data['children'].fillna(data['children'].median())
data['country'] = data['country'].fillna('Unknown')
data['agent'] = data['agent'].fillna(0)
data['company'] = data['company'].fillna(0)

# 4. Pemilihan fitur: Memilih fitur untuk clustering
features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children']

# Mengubah fitur menjadi numerik
X = data[features]

# 5. Menentukan jumlah cluster menggunakan Silhouette Score
silhouette_scores = []
range_n_clusters = list(range(2, 11))  # Menguji jumlah cluster dari 2 hingga 10

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Untuk n_clusters = {n_clusters}, Silhouette Score adalah {silhouette_avg}")

# Menentukan jumlah cluster terbaik berdasarkan Silhouette Score tertinggi
optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
print(f"Jumlah cluster optimal adalah {optimal_n_clusters} dengan Silhouette Score {max(silhouette_scores)}")

# 6. Pembuatan model: Menggunakan K-Means clustering dengan jumlah cluster optimal.
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
kmeans.fit(X)

# Menambahkan hasil cluster ke dataset
data['cluster'] = kmeans.labels_

# Define a dictionary to map cluster labels to more descriptive names
# (Disarankan untuk membuat deskripsi berdasarkan hasil yang diobservasi setelah clustering)
cluster_names = {i: f"Cluster {i+1}" for i in range(optimal_n_clusters)}

# Create a new column with descriptive cluster names
data['cluster_name'] = data['cluster'].map(cluster_names)

# 7. Evaluasi Model: Visualisasi hasil cluster
# Statistik deskriptif untuk masing-masing cluster
cluster_stats = data.groupby('cluster').agg({'lead_time': 'mean',
                                             'stays_in_weekend_nights': 'mean',
                                             'stays_in_week_nights': 'mean',
                                             'adults': 'mean',
                                             'children': 'mean'})

print(cluster_stats)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='lead_time', y='stays_in_week_nights', hue='cluster_name', palette='viridis')
plt.title('K-Means Clustering')
plt.legend(title='Cluster')
plt.show()

# Melihat perbandingan jumlah pembatalan di antara kelompok-kelompok tersebut
# Hitung jumlah pembatalan per kelompok
cancellation_counts = data.groupby('cluster')['is_canceled'].sum()

# Visualisasi perbedaan jumlah pembatalan
plt.figure(figsize=(8, 6))
sns.barplot(x=cancellation_counts.index, y=cancellation_counts.values)
plt.xlabel('Cluster')
plt.ylabel('Jumlah Pembatalan')
plt.title('Perbedaan Jumlah Pembatalan antara Kelompok-Kelompok')
plt.show()
