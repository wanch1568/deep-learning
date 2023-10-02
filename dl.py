import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# サンプルデータ
sizes = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
distances = np.array([1, 0.8, 2, 1.5, 0.5, 1.2, 2.5, 2, 0.2, 1])
rents = np.array([180000, 185000, 183000, 190000, 210000, 200000, 178000, 180000, 220000, 195000])

# 特徴量としてsizesとdistancesを結合
print(np.vstack((sizes, distances)))
X = np.vstack((sizes, distances)).T
print(X)
y = rents
# スケーラーのインスタンスを作成
scaler = StandardScaler()

# 特徴量のスケーリング
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
# 線形回帰モデルのトレーニング
model = LinearRegression()
print(X_scaled)
print(X)

# スケーリングされた特徴量でモデルをトレーニング
model.fit(X_train, y_train)

# 予測と評価
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Mean Squared Error: {mse}")
