"""
Nim AI — Q-Learning
--------------------
Built as part of CS50 AI Fundamentals — Lecture 0.

The game: several piles of stones. Players take turns removing
any number of stones from a single pile. The player who removes
the last stone loses.

The AI trains by playing thousands of games against itself.
After training, it plays against you — and it almost never loses.

Run: python nim_qlearning.py
"""

import random


class Nim:
    """represents the game state and the rules"""

    def __init__(self, piles=None):
        self.piles = piles or [1, 3, 5, 7]
        self.player = 0   # 0 or 1
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """returns all valid (pile, count) moves given the current piles"""
        actions = set()
        for i, pile in enumerate(piles):
            for n in range(1, pile + 1):
                actions.add((i, n))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action

        if self.winner is not None:
            raise Exception("game is already over")
        if pile < 0 or pile >= len(self.piles):
            raise Exception("invalid pile")
        if count < 1 or count > self.piles[pile]:
            raise Exception("invalid count")

        self.piles[pile] -= count
        self.switch_player()

        # whoever faces all-empty piles loses
        if all(p == 0 for p in self.piles):
            self.winner = self.player


class NimAI:
    """Q-Learning agent for Nim"""

    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = {}          # Q-table: (state, action) → value
        self.alpha = alpha   # learning rate
        self.epsilon = epsilon  # exploration rate

    def update(self, old_state, action, new_state, reward):
        """apply the Q-Learning update rule after each move"""
        old_q = self.get_q(old_state, action)

        # best Q-value achievable from the new state
        best_future = self.best_future_reward(new_state)

        # standard Q update
        self.q[(old_state, action)] = old_q + self.alpha * (
            reward + best_future - old_q
        )

    def get_q(self, state, action):
        """return Q-value for (state, action), defaulting to 0"""
        return self.q.get((state, action), 0)

    def best_future_reward(self, state):
        """return the highest Q-value possible from this state"""
        actions = Nim.available_actions(state)
        if not actions:
            return 0
        return max(self.get_q(state, a) for a in actions)

    def choose_action(self, state, epsilon=True):
        """
        choose an action using epsilon-greedy strategy:
        - with probability epsilon: explore (random action)
        - otherwise: exploit (best known action)
        """
        actions = list(Nim.available_actions(state))

        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        # pick action with highest Q-value, random tiebreak
        best_val = max(self.get_q(state, a) for a in actions)
        best_actions = [a for a in actions if self.get_q(state, a) == best_val]
        return random.choice(best_actions)


def train(n_games=10000):
    """let the AI play against itself for n_games to build the Q-table"""
    player = NimAI()

    for i in range(n_games):
        game = Nim()
        last = {0: {"state": None, "action": None},
                1: {"state": None, "action": None}}

        while True:
            state = tuple(game.piles)
            action = player.choose_action(state)

            # remember what this player did
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            game.move(action)
            new_state = tuple(game.piles)

            if game.winner is not None:
                # loser gets -1, winner gets +1
                player.update(state, action, new_state, -1)
                if last[game.player]["state"] is not None:
                    player.update(
                        last[game.player]["state"],
                        last[game.player]["action"],
                        new_state,
                        1
                    )
                break

            # game continues — no reward yet, update previous player's move
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    return player


def play_against_human(ai):
    """let a human play against the trained AI"""
    game = Nim()
    human_player = random.randint(0, 1)

    print()
    print("=" * 40)
    print("  Nim — Play Against the AI")
    print("=" * 40)
    print()
    print(f"  You are player {'1' if human_player == 0 else '2'}.")
    print(f"  Whoever removes the last stone loses.")
    print()

    while True:
        # show current board
        print("  Piles:")
        for i, pile in enumerate(game.piles):
            print(f"    Pile {i}: {'I ' * pile}({pile})")
        print()

        if game.player == human_player:
            print("  Your turn.")
            while True:
                try:
                    pile = int(input("  Choose pile: "))
                    count = int(input("  How many to remove: "))
                    if (pile, count) in Nim.available_actions(game.piles):
                        break
                    print("  Invalid move. Try again.")
                except ValueError:
                    print("  Enter a number.")
        else:
            print("  AI is thinking...")
            pile, count = ai.choose_action(tuple(game.piles), epsilon=False)
            print(f"  AI removes {count} from pile {pile}.")

        game.move((pile, count))
        print()

        if game.winner is not None:
            if game.winner == human_player:
                print("  You lost. The AI wins.")
            else:
                print("  You win! The AI made a mistake.")
            print()
            break


if __name__ == "__main__":
    print("=" * 40)
    print("  Nim AI — Q-Learning")
    print("  CS50 AI Fundamentals, Lecture 0")
    print("=" * 40)
    print()
    print("Training AI (10,000 games against itself)...")
    ai = train(10000)
    print("Training complete.")
    play_against_human(ai)

