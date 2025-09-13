# Day 31 – Self-Organizing Maps (SOMs)

## 📌 Overview
Self-Organizing Maps (SOMs) are **unsupervised neural networks** used for **clustering and visualization** of high-dimensional data.

## 🚀 Implementation
- Dataset: Customer Segmentation (`customers.csv`)
- Features used: Age, Annual Income, Spending Score
- Library: `MiniSom`
- Steps:
  1. Normalize dataset
  2. Train SOM on a 10x10 grid
  3. Visualize clusters using distance map

## 📊 Output
- SOM distance map showing clusters of customers
- Each customer is mapped to its "winning neuron" (cluster node)

## 🛠️ Installation
```bash
pip install minisom matplotlib pandas scikit-learn
