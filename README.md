# CS50x AI Week — Notes & Projects

Personal study repo for Harvard's CS50x AI week.
Notes written by hand during lectures, then digitized.
Every concept has a project built from scratch to make it stick.

---

## Contents

| Lecture | Topic | Notes | Projects |
|---------|-------|-------|---------|
| 0 | Reinforcement Learning | ✅ | Maze Solver, Nim AI |
| 1 | Neural Networks & Loss Functions | ✅ | Perceptron from scratch |
| 2 | Unsupervised Learning | ✅ | K-Means, Recommender System |
| 3 | — | Soon | — |
| 4 | — | Soon | — |
| 5 | — | Soon | — |
| 6 | — | Soon | — |

---

## Running the Projects

All projects are pure Python — no ML libraries needed.

```bash
# Lecture 0
python lecture-0-reinforcement-learning/maze_qlearning.py
python lecture-0-reinforcement-learning/nim_qlearning.py

# Lecture 1
python lecture-1-neural-networks/perceptron.py

# Lecture 2
python lecture-2-unsupervised/kmeans.py
python lecture-2-unsupervised/recommender.py
```

---

## What I Learned

**Lecture 0:** An agent that learns purely from reward/punishment signals can figure out optimal strategies in thousands of games. Watching the Nim AI beat you after training on nothing but experience is genuinely surprising.

**Lecture 1:** The formula `w₁n₁ + w₂n₂ + ... + b` sounds simple. Building it from scratch and watching the weights adjust toward the right answer makes it real.

**Lecture 2:** Clustering is one of those things that seems obvious until you have to explain why it works. K-Means converging in 6 iterations on 120 random points is oddly satisfying.

---

## About

Pure Python implementations of core AI/ML algorithms from scratch — Q-Learning, perceptron, K-Means, and recommender systems. No ML libraries. Just the math.
