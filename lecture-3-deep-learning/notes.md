# Lecture 3 — Deep Learning & CNNs

The jump from lecture 1 to here is big. A single perceptron worked fine for
4 inputs. But what about an image? A 28×28 grayscale photo is already 784 inputs.
Color image? Multiply by 3. Video? Now you have duration × height × width.
A fully connected network on that would have millions of weights. It doesn't scale.

This is the problem deep learning was built to solve.

---

## The Shift — Multiple Layers

Instead of one neuron doing everything, we stack layers.
Each layer learns something more abstract than the one before it.

```
Input → [Hidden 1] → [Hidden 2] → Output
 784      detects     combines     digit
pixels    edges       shapes       0-9
```

The lecture used handwritten digit recognition as the example.
We have a lot of input images and want to classify them into 10 categories (0-9).
Classic classification problem, and the go-to dataset for it is **MNIST**.

> Note: the lecture said MNIST is the most popular project for this concept.

---

## Convolutional Layer

This was the real breakthrough.

A fully connected layer connects every pixel to every neuron — which is
computationally insane for images. A convolutional layer does something smarter:

- Takes a small window (3×3 pixels)
- Applies a filter (weights) to just that window
- Slides the filter across the entire image
- Produces a **feature map**

The filter is the same wherever it slides. So the network learns to detect
a feature (like a horizontal edge) and then finds it anywhere in the image,
not just in one fixed position. That's the real insight.

### Feature Map

The output of a convolutional layer after it detects patterns.
One filter = one feature map. A layer usually runs many filters in parallel,
so you end up with a stack of feature maps.

---

## Pooling Layer

After a conv layer, the feature maps are still large. Pooling shrinks them.

A building block in CNNs used to reduce the spatial dimensions
(width and height) of feature maps.

Max pooling: take a 2×2 region, keep the highest value, throw the rest away.

```
Input → Conv layer 1 → Pooling layer 1 → Conv layer 2 → ... → Output
```

Why bother? Smaller maps = fewer parameters in the next layer = faster training.
Also helps the network care less about exactly *where* a feature appears.

---

## Images and Color

Grayscale images are width × height. That's it.

RGB adds a third dimension — one channel each for Red, Green, Blue.
So a 28×28 color image is actually 28×28×3. Convolutions handle each channel
separately and then combine them.

Video is just a sequence of frames: duration × height × width.

---

## Training vs. Test Set

How do you know if the model actually learned something, or just memorized the data?

Split the dataset:
- **Training set** (~80%) — model sees this and learns from it
- **Test set** (~20%) — model never sees this during training

Test set simulates the real world. If the model does great on training and
badly on test, something's wrong.

### Overfitting

When the model learns the training data "too well."

It memorizes noise instead of learning the actual pattern.
High training accuracy, low test accuracy. That's the signature.

Common fixes: dropout, more data, simpler model.

---

## Transfer Learning

Taking an AI model that's already smart at one big task and using it
to solve a new, related task.

A model pre-trained on millions of images already knows edges, textures, shapes.
No need to start from scratch — use those weights as a starting point.

### Fine-Tuning

A specific type of transfer learning where you continue training a pre-trained
model on your own dataset. Usually with a small learning rate, because you
don't want to destroy what it already learned.

---

## Sounds

Sounds are waves with frequency. This matters because audio can be visualized
as a spectrogram — a 2D image of frequencies over time. Which means a CNN
built for images can also work on audio.

---

## Questions I Want to Dig Into

When a 3×3 filter slides across a 28×28 image with valid padding,
how many positions does it cover? The output size shrinks — what's the formula?

The lecture said to watch the main video for the full CNN forward pass explanation.
I kept that note in because some of this is clearer in motion than on paper.

Does fine-tuning always target the last few layers, or does it depend on
how different the new task is from the original?

---

## Projects

`mnist_cnn.py` — CNN that classifies handwritten digits from MNIST.
Shows conv layers, pooling, training/test split, and overfitting in action.

---

## Source

Course: Fundamentals of Artificial Intelligence
Instructors: Brian Yu & David J. Malan — Harvard University
Video: https://www.youtube.com/watch?v=L1lQtyqpezI&list=PLJPcEQXX4i60VGmCvt1TZsprC7IGdHMpn&index=4
