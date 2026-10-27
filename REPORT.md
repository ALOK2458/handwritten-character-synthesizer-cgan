# Project Report: Handwritten Digital Font and Character Synthesizer

## Abstract

This project trains a conditional generative adversarial network (cGAN) with
PyTorch to synthesize grayscale handwritten English letters. A letter label
conditions both the generator and discriminator. Generated examples can be
viewed and downloaded in a Streamlit interface. A separate CNN is used to
estimate conditional accuracy. No training metric or image-quality result is
claimed until the scripts have actually been run.

## Problem statement

Generating a particular character is more useful than generating an
uncontrolled image. The system therefore accepts a desired A-Z class and a
random noise vector, and learns to produce varied images for that class.

## Dataset and preprocessing

The project uses torchvision's EMNIST Letters split. This dataset contains
28 x 28 grayscale character images. Its original labels are 1-26; code maps
them to 0-25. Images are corrected to ordinary reading orientation,
converted to tensors, and normalized from [0, 1] to [-1, 1]. Training and test
splits remain separate for classifier evaluation.

## Method

The generator embeds a class label, joins the embedding with random noise,
then maps the vector through fully connected layers to 784 pixels. The
discriminator embeds the label, joins it with the flattened image, and
predicts one real/fake logit. Both networks are optimized with Adam and
binary cross-entropy with logits. A standalone CNN classifier is trained on
the real training split and evaluated on the dataset test split before it is
used to score generated samples.

## Evaluation

The training program saves discriminator and generator losses, generated
sample grids, and model checkpoints. `evaluate.py` reports the fraction of
generated images that a held-out-real-data classifier predicts as the
requested letter. This is a classifier-based conditional-accuracy proxy. It
can be biased by classifier errors and does not measure style diversity,
legibility to people, or memorization. Report the actual output from your own
run; do not copy hypothetical metrics into the final submission.

## Limitations and future work

The simple fully connected architecture may produce less coherent strokes
than convolutional GANs. Classifier accuracy is not a complete image-quality
metric. EMNIST has a particular writing style distribution and the output is
not a font file or a reproduction of one person's handwriting. Possible
extensions include a convolutional GAN, FID or other distribution metrics,
human evaluation, and exporting vector glyphs after a separate tracing step.

## Reproducibility

Use the commands in README.md, record Python/PyTorch versions, hardware,
epoch counts, and the values printed by the training/evaluation scripts. A
fixed seed is provided, but exact results can vary across hardware and
library versions.
