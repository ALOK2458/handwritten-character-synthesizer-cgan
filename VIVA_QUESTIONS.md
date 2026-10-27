# Viva Questions and Answers

1. **What is a GAN?**
   A generative adversarial network trains a generator to create data and a
   discriminator to distinguish generated data from real data.

2. **Why is this model conditional?**
   Both networks receive a letter label, allowing generation to be directed
   toward a requested class.

3. **What is the role of noise?**
   Random noise gives the generator room to produce different examples for
   the same letter.

4. **Why use an embedding for labels?**
   An embedding maps each discrete class ID to a trainable feature vector that
   can be combined with image or noise features.

5. **Why normalize pixels to [-1, 1]?**
   The generator uses `tanh`, whose output range is [-1, 1], so matching the
   training-image range makes its output directly comparable to real data.

6. **What does the discriminator output?**
   One real/fake logit for an image and its supplied class; the loss applies
   the sigmoid calculation internally.

7. **What are the two training objectives?**
   The discriminator learns to score real, correctly labeled images above
   generated images. The generator learns to make generated images score as
   real for their requested labels.

8. **How is the conditional-accuracy metric calculated?**
   A separately trained CNN predicts labels for generated images. The metric
   is the fraction whose predicted label equals the requested label.

9. **Is conditional accuracy a complete measure of quality?**
   No. It depends on the classifier, can reward repetitive samples, and does
   not measure human legibility or diversity.

10. **Why is the EMNIST label mapping needed?**
    EMNIST Letters stores labels from 1 to 26, whereas PyTorch embeddings and
    cross-entropy classes conventionally use indices 0 to 25.

11. **What is mode collapse?**
    It is when the generator produces only a narrow range of outputs instead
    of diverse examples. A single conditional accuracy score may not reveal
    it.

12. **How could this project be improved?**
    Try convolutional generator/discriminator architectures, quantify
    diversity, compare against real test images, tune hyperparameters, and
    perform human evaluation.
