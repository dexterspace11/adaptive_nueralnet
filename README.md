# Adaptive NeuralNet

**AdaptiveNeuralNet** is a lightweight PyTorch-based neural network module designed for both classification and forecasting tasks. It includes automatic parameter compression and a framework for learning efficiency awareness.

---

## 🚀 Features

- 🔁 Supports **classification** and **regression/forecasting**
- 🧠 Adaptive architecture using `nn.Module`
- 🗜️ Parameter compression via magnitude-based filtering
- 🎯 Efficiency-aware learning behavior (extendable)
- 🧪 Easy-to-train with minimal configuration

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/adaptive-neuralnet.git
cd adaptive-neuralnet

# Install Dependencies
pip install -r requirements.txt

#Import the Module
from adaptive_neuralnet02 import AdaptiveNeuralNet, train_model, compress_and_report

#Classification Example
import torch
import numpy as np
from adaptive_neuralnet02 import AdaptiveNeuralNet, train_model

# Sample binary classification dataset
X = np.random.rand(100, 10)
y = (np.sum(X, axis=1) > 5).astype(float)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

model = AdaptiveNeuralNet(input_size=10, task='classification')
trained_model = train_model(model, X_tensor, y_tensor, task='classification')

#Forecasting Regression Example
import torch
import pandas as pd
import numpy as np
from adaptive_neuralnet02 import AdaptiveNeuralNet, train_model

# Load a DataFrame with a 'Close' column
df = pd.read_excel("your_data.xlsx")
close = df['Close'].values

# Generate sequences for training
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 7
X, y = create_sequences(close, seq_len)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

model = AdaptiveNeuralNet(input_size=seq_len, task='regression')
trained_model = train_model(model, X_tensor, y_tensor, epochs=100, task='regression')

#Parameter Compression
from adaptive_neuralnet02 import compress_and_report
compress_and_report(trained_model)

#Customization
task: Choose 'classification' or 'regression'

input_size: Sequence or feature length

train_model: Supports both tasks with automatic loss selection

#Forecast Output Format
Date               Forecast_Close
2025-04-01 10:00   90324.71
2025-04-02 10:00   90412.87
...

