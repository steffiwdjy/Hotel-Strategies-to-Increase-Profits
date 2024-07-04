import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import numpy as np
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

# 4. Pemilihan fitur: Menambah beberapa fitur tambahan untuk eksperimen.
features = [
    'lead_time', 'adults', 'previous_cancellations', 'booking_changes',
    'total_of_special_requests', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'children', 'babies', 'is_repeated_guest'
]
target = 'is_canceled'

# 5. Pembuatan model: Menggunakan K-Nearest Neighbors.
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Oversampling using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Standardizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning menggunakan GridSearchCV dengan pengurangan ukuran grid
param_grid = {
    'n_neighbors': [5, 13, 19],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Menampilkan parameter terbaik
best_params = grid_search.best_params_
print(f'Best params: {best_params}')

# Menggunakan model terbaik
knn = KNeighborsClassifier(**best_params)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 6. Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Canceled', 'Canceled'], yticklabels=['Not Canceled', 'Canceled'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for K-Nearest Neighbors')
plt.show()
