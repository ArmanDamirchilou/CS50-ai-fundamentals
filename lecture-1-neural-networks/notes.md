# Lecture 1 — Neural Networks & Loss Functions

> My notes from CS50x AI Week — machine learning section.
> The big shift from lecture 0: instead of learning from rewards, we learn from labeled data.

---

## Loss Functions

How do we know if our predictions are any good?
Loss functions measure the gap between what we predicted and what was actually correct.

### Absolute Error

```
|ŷ - y|
```

Simple distance between prediction and reality. Always positive.

### Squared Error

```
(ŷ - y)²
```

Penalizes big mistakes much more than small ones.
A prediction that's off by 10 gets 100× the penalty of one that's off by 1.

### Mean Squared Error (MSE)

Average squared error across all training examples.
This is what we want to minimize during training.

```
MSE = (1/n) × Σ(ŷᵢ - yᵢ)²
```

*The goal of training = drive MSE as low as possible.*

---

## Regression vs. Classification

| | Regression | Classification |
|---|---|---|
| Output | A number | A category |
| Example | Predict house price | Cat or dog? |
| Loss function | MSE | Cross-entropy |

**Classification:** given inputs, learn features, then choose what's correct.

---

## Neural Networks

Inspired by the human brain — made up of neurons connected to each other.

We're not building biological neural networks. We build **artificial** ones.
The math is similar; the substrate is very different.

---

## Perceptron (Single Neuron)

The simplest neural network — just one neuron.

```
Temperature  (n₁) ──[w₁]──┐
Air Pressure (n₂) ──[w₂]──┤
Cloud Cover  (n₃) ──[w₃]──┼──→ [Σ + b] ──→ activation ──→ Rain? (y)
Humidity     (n₄) ──[w₄]──┘
```

**Formula:**

```
output = activation(w₁n₁ + w₂n₂ + w₃n₃ + w₄n₄ + b)
```

Each input `nᵢ` has a weight `wᵢ` and there's a bias `b`.
The bias shifts the decision boundary — without it the model is too constrained.

*Note: each of these n's has a weight and impact on the output*

---

## Activation Functions

How should the neuron decide whether to "fire" or not?

### 1. Unit Step Function

```
if x < 0  → output 0
if x ≥ 0  → output the input (1)
```

Binary — either fires or doesn't. Good for simple yes/no decisions.

### 2. ReLU (Rectified Linear Unit)

```
if x < 0  → output 0
if x ≥ 0  → output x (the actual value)
```

More flexible than unit step. Lets through positive values as-is.
Most common activation function in modern deep learning.

---

## Hidden Layer

A layer of neurons between input and output.

> Hidden Layer = extracts features and finds patterns in your data

Instead of the inputs connecting directly to the output,
they first pass through hidden layers that learn intermediate representations.

**Example:** classifying handwritten digits
- Layer 1 might learn edges and curves
- Layer 2 combines those into shapes
- Output layer classifies the digit

---

## Simple Math Example

```
inputs: 5, 4
weights: 2, 1
bias: 3

weighted sum: (4×1) + (5×2) + 3 = 4 + 10 + 3 = 17
```

---

## Key Takeaways

1. **Loss functions quantify how wrong we are.** Without a loss function, we have no signal to learn from. MSE is the most common starting point.

2. **Weights and bias are what the model learns.** The architecture (perceptron, hidden layers) stays fixed. Training adjusts weights and bias until predictions are accurate.

3. **Activation functions introduce non-linearity.** Without them, stacking more layers doesn't help — the whole network collapses to a single linear function.

---

## My Questions

1. Why does ReLU work better than unit step in deep networks? What happens during training with a step function?

2. How many neurons should a hidden layer have? Is there a rule, or is it trial and error?

3. The formula `w₁n₁ + w₂n₂ + ... + b` is linear. How do neural networks learn non-linear patterns?

---

## Project Built from This Lecture

`perceptron.py` — single perceptron from scratch, no libraries.
Trains on synthetic weather data, shows loss curve, outputs learned weights.

---

## Source

Lecture: CS50x — Introduction to Artificial Intelligence

Instructor: David J. Malan

Video: https://www.youtube.com/watch?v=oHuvUkMPffI&list=PLJPcEQXX4i60VGmCvt1TZsprC7IGdHMpn&index=2
