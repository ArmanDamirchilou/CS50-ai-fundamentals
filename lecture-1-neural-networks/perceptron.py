"""
Perceptron from Scratch
-----------------------
After lecture 1, I wanted to actually implement the formula
from the board: w1*n1 + w2*n2 + w3*n3 + w4*n4 + b

No libraries. Just the math. Predicts rain or no-rain based on
temperature, air pressure, cloud cover, and humidity.
"""

import random


FEATURES = ["temperature", "air_pressure", "cloud_cover", "humidity"]


def unit_step(x):
    # less than 0 → output 0
    # at least 0 → output 1
    # this is the one from lecture — simplest possible activation
    return 1 if x >= 0 else 0


def relu(x):
    # rectified linear unit
    # less than 0 → 0, at least 0 → keep the value as-is
    # smoother than unit step, tends to train better in practice
    return max(0.0, x)


def predict(inputs, weights, bias, activation="step"):
    # w1*n1 + w2*n2 + w3*n3 + w4*n4 + b — exactly the lecture formula
    total = bias
    for x, w in zip(inputs, weights):
        total += x * w

    if activation == "step":
        return unit_step(total)
    return relu(total)


def absolute_error(predicted, actual):
    # |ŷ - y| — from the notes
    return abs(predicted - actual)


def squared_error(predicted, actual):
    # (ŷ - y)² — penalizes big mistakes way more than small ones
    return (predicted - actual) ** 2


def mean_squared_error(predictions, actuals):
    # average squared error across all examples
    # the number we want to drive down during training
    total = sum(squared_error(p, a) for p, a in zip(predictions, actuals))
    return total / len(predictions)


def train(data, labels, epochs=100, lr=0.1):
    # start with random small weights — if we start at zero
    # all weights update identically and that's a problem
    weights = [random.uniform(-0.5, 0.5) for _ in FEATURES]
    bias = 0.0
    history = []

    for epoch in range(epochs):
        preds, actuals = [], []

        for inputs, label in zip(data, labels):
            pred = predict(inputs, weights, bias)
            error = label - pred

            # nudge each weight in the right direction
            # bigger error = bigger nudge (scaled by learning rate)
            for i in range(len(weights)):
                weights[i] += lr * error * inputs[i]
            bias += lr * error

            preds.append(pred)
            actuals.append(label)

        mse = mean_squared_error(preds, actuals)
        history.append(mse)

        if (epoch + 1) % 20 == 0:
            acc = sum(p == a for p, a in zip(preds, actuals)) / len(labels)
            print(f"  epoch {epoch+1:3d} | loss: {mse:.4f} | acc: {acc*100:.1f}%")

    return weights, bias, history


def make_weather_data(n=300):
    # synthetic data — pattern: high humidity + low pressure + cloudy = rain
    # added noise so the perceptron actually has to learn something
    data, labels = [], []
    for _ in range(n):
        temp = random.uniform(0, 1)
        pressure = random.uniform(0, 1)
        clouds = random.uniform(0, 1)
        humidity = random.uniform(0, 1)

        score = humidity * 0.5 + (1 - pressure) * 0.3 + clouds * 0.2
        label = 1 if score + random.uniform(-0.1, 0.1) > 0.5 else 0

        data.append([temp, pressure, clouds, humidity])
        labels.append(label)
    return data, labels


def loss_curve(history):
    print("\n  Training loss over time (lower = better):\n")
    peak = max(history) or 1
    for i, val in enumerate(history):
        if (i + 1) % 10 == 0:
            bar = "█" * int(val / peak * 30)
            print(f"  ep {i+1:3d} | {bar:<30} {val:.4f}")


if __name__ == "__main__":
    print("=" * 45)
    print("  Perceptron — Lecture 1 | CS50x AI Week")
    print("=" * 45)
    print()

    data, labels = make_weather_data(300)
    split = int(len(data) * 0.8)

    print(f"Training on {split} samples...")
    print()
    weights, bias, history = train(
        data[:split], labels[:split], epochs=100, lr=0.05
    )

    test_preds = [predict(x, weights, bias) for x in data[split:]]
    correct = sum(p == a for p, a in zip(test_preds, labels[split:]))
    total = len(labels[split:])
    print(f"\nTest: {correct}/{total} correct ({correct/total*100:.1f}%)")

    loss_curve(history)

    print("\n  What the model learned (weight per feature):")
    for f, w in zip(FEATURES, weights):
        direction = "→ rain" if w > 0 else "→ no rain"
        print(f"    {f:<15} {w:+.3f}  {direction}")

