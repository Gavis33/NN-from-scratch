# 🧠 NN from Scratch

A **Neural Network implemented from scratch** with nothing but **NumPy**.  
This project demystifies how deep learning works under the hood — no TensorFlow, no PyTorch, just math, matrix multiplications, and code.  

---

## 🚀 Features
- Custom implementation of:
  - Forward Propagation
  - Backward Propagation
  - Gradient Descent Optimization
  - Activation Functions (ReLU, Softmax, etc.)
- Trains on the **MNIST dataset** (handwritten digits 0–9)
- Accuracy tracking per epoch
- Clean modular code for easy experimentation

---

## 📂 Project Structure
```
NN-from-scratch/
│── data/                # Dataset (MNIST handled via torchvision or numpy)
│── nn.py                # Core neural network implementation
│── utils.py             # Helper functions (activations, accuracy, etc.)
│── train.py             # Training loop with gradient descent
│── README.md            # You’re here
```

---

## ⚡ How It Works
1. **Initialize parameters** (weights & biases for each layer)
2. **Forward propagation** to compute predictions
3. **Backward propagation** to compute gradients
4. **Update parameters** using gradient descent
5. Repeat until the network learns to classify digits 🎯

---

## 🏃 Usage
Clone this repo:
```bash
git clone https://github.com/yourusername/NN-from-scratch.git
cd NN-from-scratch
```

Install dependencies:
```bash
pip install numpy matplotlib
```

Train the model:
```bash
python train.py
```

---

## 📊 Results
- Trains on **MNIST (60k training images, 10k test images)**
- Achieves **~90%+ accuracy** with just a simple 2-layer NN

---

## 🎯 Motivation
Most tutorials rely on PyTorch or TensorFlow.  
This repo strips things down to the **raw math + code**, so you truly understand:
- How gradients flow
- Why activation functions matter
- How optimization works

---

## 🌟 Future Work
- Add more layers (deep NN)
- Implement different optimizers (SGD, Adam)
- Experiment with other datasets (CIFAR-10, Fashion-MNIST)
