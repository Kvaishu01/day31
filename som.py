import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("customers.csv")
X = data[["Age", "Annual_Income", "Spending_Score"]].values

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize SOM (10x10 grid, 3 features)
som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(X_scaled)

# Train SOM
print("Training SOM...")
som.train_random(X_scaled, num_iteration=200)

# Plot distance map
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='coolwarm')
plt.colorbar()

# Mark data points
for i, x in enumerate(X_scaled):
    w = som.winner(x)
    plt.text(w[0] + 0.5, w[1] + 0.5, str(i + 1), fontsize=7, color='black')

plt.title("Customer Segmentation using SOM")
plt.show()
