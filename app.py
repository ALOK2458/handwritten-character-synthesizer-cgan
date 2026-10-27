"""Streamlit interface for generating and downloading EMNIST-style letters."""

from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from cgan import CharacterClassifier, Generator, NUM_CLASSES
from data import LETTERS, letter_to_label, label_to_letter


st.set_page_config(page_title="Handwritten Letter Synthesizer")
st.title("Handwritten Letter Synthesizer")
st.write(
    "Generate samples from a cGAN trained on EMNIST Letters. "
    "Each seed gives a different sample; it does not represent a verified person's handwriting."
)

checkpoint_path = Path("checkpoints/cgan_latest.pt")
classifier_path = Path("checkpoints/classifier.pt")
if not checkpoint_path.is_file():
    st.error("No trained GAN checkpoint found. Follow the training steps in README.md first.")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
if checkpoint.get("num_classes") != NUM_CLASSES:
    st.error("This checkpoint is not a 26-class EMNIST Letters model.")
    st.stop()
noise_size = int(checkpoint["noise_size"])
generator = Generator(noise_size=noise_size).to(device)
generator.load_state_dict(checkpoint["generator_state_dict"])
generator.eval()

classifier = None
if classifier_path.is_file():
    classifier_checkpoint = torch.load(classifier_path, map_location=device)
    classifier = CharacterClassifier().to(device)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"])
    classifier.eval()

selected_letter = st.selectbox("Choose a letter", LETTERS)
number_of_styles = st.slider("Number of samples", min_value=1, max_value=16, value=4)
seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42)

if st.button("Generate handwriting"):
    label = letter_to_label(selected_letter)
    labels = torch.full((number_of_styles,), label, dtype=torch.long, device=device)
    random_generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        number_of_styles,
        noise_size,
        generator=random_generator,
        device=device,
    )
    with torch.no_grad():
        images = generator(noise, labels).cpu()
        predictions = classifier(images.to(device)).softmax(dim=1) if classifier else None

    samples = []
    for index, image_tensor in enumerate(images):
        pixels = (
            image_tensor.squeeze(0).mul(0.5).add(0.5).clamp(0, 1).mul(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        image = Image.fromarray(pixels, mode="L")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        prediction = None
        if predictions is not None:
            predicted_label = int(predictions[index].argmax().item())
            prediction = (
                label_to_letter(predicted_label),
                float(predictions[index, predicted_label].item()),
            )
        samples.append({"png": buffer.getvalue(), "prediction": prediction})
    st.session_state["generated_samples"] = {
        "letter": selected_letter,
        "seed": int(seed),
        "samples": samples,
    }

saved_samples = st.session_state.get("generated_samples")
if saved_samples:
    columns = st.columns(4)
    for index, sample in enumerate(saved_samples["samples"]):
        image = Image.open(BytesIO(sample["png"]))
        with columns[index % len(columns)]:
            st.image(image, caption=f"Sample {index + 1}", width=140)
            if sample["prediction"] is not None:
                predicted_letter, confidence = sample["prediction"]
                st.caption(
                    f"Classifier: {predicted_letter} "
                    f"({confidence:.1%} confidence; not a quality score)"
                )
            st.download_button(
                label=f"Download sample {index + 1}",
                data=sample["png"],
                file_name=f"letter_{saved_samples['letter']}_{index + 1}.png",
                mime="image/png",
                key=f"download_{saved_samples['letter']}_{saved_samples['seed']}_{index}",
            )
