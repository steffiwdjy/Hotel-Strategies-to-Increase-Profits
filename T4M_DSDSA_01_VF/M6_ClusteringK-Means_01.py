from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Memuat dataset
data = pd.read_csv('C://mak/DSDS/T4/hotel_bookings.csv')

# Menangani missing values jika ada
data = data.dropna()

# Membuat fitur dengan interaksi tambahan
data['lead_time_stays_in_week_nights'] = data['lead_time'] * data['stays_in_week_nights']
data['adults_children'] = data['adults'] * data['children']
data['total_stays'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']

# Memilih fitur untuk clustering
features_clustering = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
                       'lead_time_stays_in_week_nights', 'adults_children', 'total_stays']
X_clustering = data[features_clustering]

# Standarisasi fitur
scaler = StandardScaler()
X_clustering = scaler.fit_transform(X_clustering)

# Menerapkan t-SNE untuk reduksi dimensi
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_clustering)

# Evaluasi dengan k=3
kmeans_3 = KMeans(n_clusters=3, random_state=42)
kmeans_3.fit(X_tsne)
clusters_3 = kmeans_3.predict(X_tsne)
silhouette_avg_3 = silhouette_score(X_tsne, clusters_3)
print(f'Silhouette Score untuk k=3 setelah t-SNE: {silhouette_avg_3}')

# Visualisasi cluster untuk k=3
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters_3, cmap='viridis')
plt.title('K-Means Clustering dengan k=3 setelah t-SNE')
plt.xlabel('Komponen t-SNE 1')
plt.ylabel('Komponen t-SNE 2')
plt.show()

# Tambahkan label cluster ke dalam DataFrame asli
data['cluster'] = clusters_3

# Reorder clusters
data['cluster'] = data['cluster'].map({0: 0, 1: 2, 2: 1})

# Warna untuk setiap cluster
palette = sns.color_palette('viridis', n_colors=3)

# Visualisasikan distribusi lead_time bagi semua cluster dengan warna yang sesuai
sns.boxplot(x='cluster', y='lead_time', data=data, palette=palette, order=[0, 1, 2])
plt.title('Distribution of Lead Time Across Clusters')
plt.show()

# Visualisasikan distribusi stays_in_weekend_nights bagi semua cluster dengan warna yang sesuai
sns.boxplot(x='cluster', y='stays_in_weekend_nights', data=data, palette=palette, order=[0, 1, 2])
plt.title('Distribution of Stays in Weekend Nights Across Clusters')
plt.show()

# Visualisasikan distribusi total_stays bagi semua cluster dengan warna yang sesuai
sns.boxplot(x='cluster', y='total_stays', data=data, palette=palette, order=[0, 1, 2])
plt.title('Distribution of Total Stays Across Clusters')
plt.show()
