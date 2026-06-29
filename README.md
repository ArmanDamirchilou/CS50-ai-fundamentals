# CS50x AI — Notes & Projects

Lecture notes and Python implementations of core AI/ML concepts.
Notes written by hand during lectures, then cleaned up here.
Every concept has at least one project built from scratch.

---

## Contents

| Lecture | Topic | Notes | Projects |
|---------|-------|-------|---------|
| 0 | Reinforcement Learning | ✅ | Maze Solver, Nim AI |
| 1 | Neural Networks & Loss Functions | ✅ | Perceptron |
| 2 | Unsupervised Learning | ✅ | K-Means, Recommender System |
| 3 | Deep Learning & CNNs | ✅ | MNIST Classifier |
| 4 | — | Soon | — |
| 5 | — | Soon | — |
| 6 | — | Soon | — |

---

## Running the Projects

Lectures 0–2 need only the Python standard library.
Lecture 3 needs PyTorch:

```bash
pip install torch torchvision
```

```bash
# Lecture 0
python lecture-0-reinforcement-learning/maze_qlearning.py
python lecture-0-reinforcement-learning/nim_qlearning.py

# Lecture 1
python lecture-1-neural-networks/perceptron.py

# Lecture 2
python lecture-2-unsupervised/kmeans.py
python lecture-2-unsupervised/recommender.py

# Lecture 3
python lecture-3-deep-learning/mnist_cnn.py
```

---

## What I Learned

**Lecture 0:** An agent that learns purely from reward and punishment can figure out optimal strategies across thousands of games. Watching the Nim AI beat you after training on nothing but experience is genuinely surprising.

**Lecture 1:** The formula `w₁n₁ + w₂n₂ + ... + b` sounds simple. Building it from scratch and watching the weights adjust toward the right answer makes it real.

**Lecture 2:** Clustering seems obvious until you have to explain why it works. K-Means converging in 6 iterations on 120 random points is oddly satisfying.

**Lecture 3:** A fully connected network on a 28×28 image would have millions of weights. Convolutional layers solve that by learning local patterns and sliding them across the whole image. The model hits 99% accuracy on digits it's never seen.
---

## About

Pure Python implementations of core AI/ML algorithms — Q-Learning, perceptrons, K-Means, CNNs. No libraries except PyTorch for the neural network projects.
