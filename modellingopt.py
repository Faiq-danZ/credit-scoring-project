import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

# 1. Sambungkan ke server MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# 2. Set nama eksperimen yang sama agar bisa dibandingin di dasbor
mlflow.set_experiment("Latihan Credit Scoring")

# 3. Baca data
data = pd.read_csv("train_pca.csv")

# 4. Bagi data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    random_state=42,
    test_size=0.2
)
input_example = X_train[0:5]

# 5. Menentukan jangkauan bumbu parameter yang mau dicoba (Metode Grid/Random Search)
# np.linspace(10, 1000, 5) artinya bikin 5 pilihan angka dari 10 sampai 1000 secara adil
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)  
max_depth_range = np.linspace(1, 50, 5, dtype=int)      

best_accuracy = 0
best_params = {}

# 6. Looping otomatis (Mencoba 5 x 5 = 25 kombinasi bumbu secara berurutan)
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        
        # Berikan nama run dinamis berdasarkan bumbu yang sedang diuji
        with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
            mlflow.autolog()

            # Latih model dengan bumbu saat ini
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Hitung akurasi
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)

            # SELEKSI ALAM: Jika kombinasi ini adalah yang terbaik, simpan modelnya ke MLflow!
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example
                )