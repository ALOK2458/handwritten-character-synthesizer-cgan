"""Fast offline tests for model shapes and EMNIST label conversions."""

import unittest

import torch

from cgan import CharacterClassifier, Discriminator, Generator
from data import label_to_letter, letter_to_label


class ProjectSmokeTests(unittest.TestCase):
    def test_letter_label_round_trip(self) -> None:
        for label in range(26):
            self.assertEqual(letter_to_label(label_to_letter(label)), label)

    def test_letter_validation(self) -> None:
        with self.assertRaises(ValueError):
            letter_to_label("AA")
        with self.assertRaises(ValueError):
            label_to_letter(26)

    def test_generator_discriminator_shapes(self) -> None:
        generator = Generator()
        discriminator = Discriminator()
        images = generator(torch.randn(4, 100), torch.tensor([0, 1, 2, 25]))
        scores = discriminator(images, torch.tensor([0, 1, 2, 25]))
        self.assertEqual(tuple(images.shape), (4, 1, 28, 28))
        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(torch.all(images >= -1).item())
        self.assertTrue(torch.all(images <= 1).item())

    def test_classifier_output_shape(self) -> None:
        logits = CharacterClassifier()(torch.randn(3, 1, 28, 28))
        self.assertEqual(tuple(logits.shape), (3, 26))


if __name__ == "__main__":
    unittest.main()
