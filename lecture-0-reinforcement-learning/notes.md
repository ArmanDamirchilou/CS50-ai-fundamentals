# Lecture 0 — Reinforcement Learning & Q-Learning

> These are my personal notes from CS50's Fundamentals of AI.
> I find RL fascinating because the agent literally learns the same way we do — by trying things and seeing what happens.

---

## What is Reinforcement Learning?

Reinforcement Learning (RL) is learning **by experience**.

Unlike supervised learning where you give the model the right answers, in RL the agent figures things out on its own. It takes actions, gets feedback (rewards or punishments), and gradually learns what works.

Think of it like training a dog. You don't explain the rules — you just reward good behavior and correct bad behavior until the dog figures out the pattern.

---

## Key Concepts

### Evaluation Function

A function that **estimates the value of a game state** without needing to play the game all the way to the end.

Why do we need this? Because in complex games like chess, you can't explore every possible future move. The evaluation function gives you a shortcut — a score that says "this position looks good" or "this position looks bad."

---

### Explore vs. Exploit

This is one of the core tensions in RL:

| | What it means | Risk |
|---|---|---|
| **Explore** | Try new actions you haven't tried before | Might waste time on bad paths |
| **Exploit** | Use what you already know works | Might miss something better |

A good RL agent needs to balance both. Too much exploration = slow learning. Too much exploitation = stuck in a local optimum.

*Example: You found a restaurant you like. Do you keep going there (exploit) or try new places (explore)?*

---

### Q-Learning

Q-Learning is a type of Reinforcement Learning. The "Q" stands for **Quality** — the quality of taking a certain action in a certain state.

Every Q-Learning setup has three things:

```
State   → where the agent currently is
Action  → what the agent can do
Reward  → feedback after taking an action
         (positive = good, negative = punishment)
```

The agent builds a **Q-table** that maps every (state, action) pair to a value. Over thousands of iterations, these values get more and more accurate.

**The core update rule:**

```
Q(state, action) = Q(state, action) + α × [reward + γ × max(Q(next_state)) - Q(state, action)]
```

Where:
- `α` (alpha) = learning rate — how fast we update
- `γ` (gamma) = discount factor — how much we care about future rewards

---

### Simulation

Running many iterations of the environment to let the agent practice. The agent doesn't learn from one experience — it needs thousands of simulations to build reliable Q-values.

---

## Vocabulary from This Lecture

| English | Meaning |
|---|---|
| Navigate | حرکت کردن — finding a path through an environment |
| Obstacle | مانع — something blocking the path |
| Punishment | جریمه — negative reward for a bad action |
| Direction | جهت — which way to move |
| Approach | رویکرد — the method or strategy used |
| Simulation | شبیه‌سازی — running the environment repeatedly |
| Evaluation | ارزیابی — scoring how good a state is |

---

## Key Takeaways

1. **RL agents learn from experience, not labels.** There's no dataset — the agent generates its own training data by interacting with the environment.

2. **The Explore vs. Exploit tradeoff is fundamental.** Every RL system has to decide how much to try new things vs. use known good strategies.

3. **Q-Learning stores knowledge in a table.** Each (state, action) pair gets a value that improves over time through the update rule.

---

## My Questions

1. How does Q-Learning scale to environments with millions of possible states? (Hint: this is why Deep Q-Learning and neural networks were invented)

2. What happens if the reward signal is delayed? For example, in chess, you only know if you won at the very end — how does Q-Learning assign credit to earlier moves?

3. Is the evaluation function in chess hand-crafted or learned? How does AlphaZero handle this differently from traditional chess engines?

---

## Project Built from This Lecture

**Maze Solver using Q-Learning** → `maze_qlearning.py`

An agent learns to navigate from start to finish in a maze, avoiding walls and obstacles. No hardcoded path — it discovers the optimal route purely through trial and error.

