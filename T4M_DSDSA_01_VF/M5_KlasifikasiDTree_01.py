import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

'''
Faktor apa saja yang sangat mempengaruhi pembatalan booking?

Model dapat mengidentifikasi fitur apa saja yang penting untuk memprediksi pembatalan
yang dapat membantu pihak hotel memahami apa saja motivasi di belakang pembatalan booking.
'''
# Load dataset yang akan digunakan
data = pd.read_csv('C://mak/DSDS/T4/hotel_bookings.csv')

# Drop missing values
data = data.dropna()

# Interactive Feature selection (Membuat fitur baru dari yang sudah ada, semacam feature engineering)
data['lead_time_stays_in_week_nights'] = data['lead_time'] * data['stays_in_week_nights']
data['adults_children'] = data['adults'] * data['children']

# Fitur tambahan yang akan digunakan
additional_features = ['arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'meal', 
                       'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 
                       'booking_changes', 'deposit_type', 'agent', 'company', 'days_in_waiting_list', 
                       'customer_type', 'adr', 'required_car_parking_spaces', 'total_of_special_requests', 
                       'reservation_status', 'reservation_status_date']

# Update fitur yang ada dengan hasil interactive feature selection
features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 
            'lead_time_stays_in_week_nights', 'adults_children'] + additional_features
X = data[features]

# Convert tipe data kategorikal menjadi dummy
X = pd.get_dummies(X)

# Definisikan target yang dicari (status pembatalan)
y = data['is_canceled']

# Split data menjadi train-test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisasikan fitur-fitur (untuk mengatasi dataset yang imbalance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Aplikasikan SMOTE ke data training (mengatasi masalah imbalance data)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Definisikan grid parameter (Bagian dari parameter tuning seperti kriteria penilaian, kedalaman pohon, dll.)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'class_weight': [None, 'balanced', {0: 1, 1: 10}, {0: 1, 1: 20}, {0: 1, 1: 50}]
}

# Inisialisasikan model Decision Tree
dt = DecisionTreeClassifier(random_state=42)

# Inisialisasi Grid Search
# untuk menemukan kombinasi parameter yang optimal untuk suatu model
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1')

# Melakukan fit Grid Search dengan data yang telah di-resample
grid_search.fit(X_resampled, y_resampled)

# Ambil dan simpan model terbaik dari hasil Grid Search
best_dt = grid_search.best_estimator_

# Prediksi data test
y_pred = best_dt.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for Optimized Decision Tree Classifier with Additional Features and SMOTE')
plt.show()

# Melakukan cross-validation untuk mengevaluasi kinerja model machine learning
# secara lebih robust dan menghindari overfitting
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(best_dt, X_resampled, y_resampled, cv=5, scoring='f1')

print(f'Cross-Validation F1 Scores: {cv_scores}')
print(f'Mean F1 Score: {cv_scores.mean()}')
print(f'Standard Deviation of F1 Scores: {cv_scores.std()}')