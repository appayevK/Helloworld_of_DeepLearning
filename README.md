# MNIST Neural Network from Scratch (NumPy)

A feedforward neural network for MNIST digit classification, implemented from scratch in pure NumPy — no PyTorch, TensorFlow, or autograd. Forward pass, backpropagation, and mini-batch gradient descent are all written by hand.

Inspired by 3Blue1Brown's neural networks series.

## Results

| Split | Accuracy |
|-------|----------|
| Training (20k samples, 3 epochs) | ~99.9% |
| Test (10k samples) | ~95.1% |

## Architecture

```
Input (784)  →  Hidden 1 (128, ReLU)  →  Hidden 2 (64, ReLU)  →  Output (10, Softmax)
```

- **Loss:** Cross-entropy
- **Optimizer:** Mini-batch SGD (batch size 32, learning rate 0.1)
- **Epochs:** 15 (did 12 then 3 more)
- **Weight init:** Uniform `[-0.5, 0.5]`, biases zero

## Implementation Details

### Forward pass
Each hidden layer computes `a = ReLU(Wx + b)`. The output layer uses softmax with the standard max-subtraction trick for numerical stability.

### Backpropagation
Gradients are derived analytically using the chain rule:

- **Output layer:** `dC/dz = ŷ - y` (the clean softmax + cross-entropy combined gradient)
- **Hidden layers:** `dC/dz = (Wᵀ · δ) ⊙ ReLU'(z)`
- **Weights:** `dC/dW = δ · aᵀ`
- **Biases:** `dC/db = δ`

Per-example gradients are accumulated across each mini-batch and averaged before the parameter update.

## Dataset

Google Colab's preloaded MNIST CSVs:
- `mnist_train_small.csv` — 20,000 training samples
- `mnist_test.csv` — 10,000 test samples

Pixel values are normalized to `[0, 1]` by dividing by 255.

## Usage

Open the notebook in Colab (or any Jupyter environment with NumPy and Pandas) and run the cells in order:

1. Load and normalize the data
2. Initialize the weight matrices and bias vectors
3. Train with `training(input, label)`
4. Evaluate on the held-out test set with `validate(x_testn, y_testn)`

## File Structure

```
.
├── mnistinwithNP.ipynb   # main notebook
└── README.md
```

## Notes

Built as a learning exercise to internalize how backpropagation actually works at the matrix level — every shape, every gradient, every update step is explicit. No frameworks, no shortcuts. p.s(in the trainer change the epoch number for 15 for the same result)
