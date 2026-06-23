"""
Maze Solver using Q-Learning
-----------------------------
Built as part of CS50 AI Fundamentals — Lecture 0.

The agent starts at 'S', needs to reach 'G', and learns to
avoid walls '#' through trial and error. No hardcoded path.
It figures everything out on its own.

Run: python maze_qlearning.py
"""

import random
import time
import os

# the maze — 0 is open, 1 is wall
MAZE = [
    [0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
]

ROWS = len(MAZE)
COLS = len(MAZE[0])
START = (0, 0)
GOAL  = (4, 6)

# actions: up, down, left, right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]

# hyperparameters — these affect how fast and well the agent learns
ALPHA   = 0.1    # learning rate: how much we update Q on each step
GAMMA   = 0.9    # discount factor: how much we care about future rewards
EPSILON = 0.3    # exploration rate: probability of trying a random action
EPISODES = 3000  # how many full runs the agent does


def is_valid(row, col):
    """check if a position is inside the maze and not a wall"""
    return 0 <= row < ROWS and 0 <= col < COLS and MAZE[row][col] == 0


def get_reward(state):
    """the agent gets a big reward for reaching the goal,
    and a small penalty for every other step to encourage efficiency"""
    if state == GOAL:
        return 100
    return -1  # small penalty per step — agent learns to be fast


def train():
    """run Q-Learning for a fixed number of episodes.
    returns the Q-table after training is done."""

    # Q-table starts at zero for everything
    q = {}
    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r][c] == 0:
                q[(r, c)] = [0.0] * len(ACTIONS)

    for episode in range(EPISODES):
        state = START

        # decay epsilon over time so agent explores less as it gets smarter
        eps = max(0.05, EPSILON - (episode / EPISODES) * 0.25)

        steps = 0
        while state != GOAL and steps < 200:
            # explore or exploit
            if random.random() < eps:
                action_idx = random.randint(0, len(ACTIONS) - 1)
            else:
                action_idx = q[state].index(max(q[state]))

            dr, dc = ACTIONS[action_idx]
            next_state = (state[0] + dr, state[1] + dc)

            if not is_valid(*next_state):
                # hit a wall — stay in place and get punished
                next_state = state
                reward = -5
            else:
                reward = get_reward(next_state)

            # Q-Learning update rule
            best_next = max(q[next_state]) if next_state in q else 0
            q[state][action_idx] += ALPHA * (
                reward + GAMMA * best_next - q[state][action_idx]
            )

            state = next_state
            steps += 1

        if (episode + 1) % 500 == 0:
            print(f"  episode {episode + 1}/{EPISODES} done")

    return q


def solve(q):
    """use the trained Q-table to find the optimal path.
    returns the list of states from start to goal."""
    state = START
    path = [state]
    visited = {state}

    while state != GOAL:
        if state not in q:
            print("stuck — no Q-values for this state")
            return None

        # always take the best known action (no exploration here)
        action_idx = q[state].index(max(q[state]))
        dr, dc = ACTIONS[action_idx]
        next_state = (state[0] + dr, state[1] + dc)

        if not is_valid(*next_state) or next_state in visited:
            print("agent got stuck in a loop — try more training episodes")
            return None

        visited.add(next_state)
        path.append(next_state)
        state = next_state

        if len(path) > ROWS * COLS:
            print("path too long — something went wrong")
            return None

    return path


def display(path):
    """print the maze with the solution path shown as dots"""
    path_set = set(path)
    symbols = {
        START: "S",
        GOAL:  "G",
    }

    print()
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            pos = (r, c)
            if pos in symbols:
                row_str += f" {symbols[pos]} "
            elif MAZE[r][c] == 1:
                row_str += " █ "
            elif pos in path_set:
                row_str += " · "
            else:
                row_str += "   "
        print(row_str)
    print()


def animate(path):
    """show the agent moving through the maze step by step"""
    for i, current in enumerate(path):
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 30)
        print(f"  Maze Solver — Q-Learning")
        print(f"  Step {i + 1} of {len(path) - 1}")
        print("=" * 30)

        past = set(path[:i])
        print()
        for r in range(ROWS):
            row_str = ""
            for c in range(COLS):
                pos = (r, c)
                if pos == current:
                    row_str += " @ "     # agent's current position
                elif pos == GOAL:
                    row_str += " G "
                elif pos == START and i > 0:
                    row_str += " · "
                elif MAZE[r][c] == 1:
                    row_str += " █ "
                elif pos in past:
                    row_str += " · "
                else:
                    row_str += "   "
            print(row_str)
        print()

        if current == GOAL:
            print(f"  ✓ Reached the goal in {len(path) - 1} steps!")
        time.sleep(0.3)


if __name__ == "__main__":
    print("=" * 30)
    print("  Maze Solver — Q-Learning")
    print("  CS50 AI Fundamentals, Lecture 0")
    print("=" * 30)
    print()
    print("Training the agent...")
    print()

    q_table = train()

    print()
    print("Training complete. Finding optimal path...")
    path = solve(q_table)

    if path:
        print(f"Solved in {len(path) - 1} steps.\n")
        print("Optimal path:")
        display(path)

        answer = input("Animate the solution? (y/n): ").strip().lower()
        if answer == "y":
            animate(path)
    else:
        print("Could not find a path. Try increasing EPISODES.")

