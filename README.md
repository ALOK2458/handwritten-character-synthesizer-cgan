# Handwritten Digital Font & Character Synthesizer (cGAN)

A beginner-friendly Python/PyTorch project that learns to generate
**handwritten-style A-Z images** from the EMNIST Letters dataset. It includes
real training and evaluation scripts, saved checkpoints, visualizations, and
a Streamlit app. It does **not** ship with fabricated metrics or pretrained
weights: run training before using generation.

> This produces individual grayscale character images, not an installable
> font file. Generated writing is dataset-like and should not be represented
> as the handwriting of a particular person.

## Project files

| File | Purpose |
|---|---|
| `cgan.py` | Conditional generator, discriminator, and evaluation classifier |
| `data.py` | EMNIST loading, letter labels, image orientation and normalization |
| `visualize_data.py` | Plot real data samples |
| `train.py` | Train cGAN; save checkpoints, generated grids, and losses |
| `train_classifier.py` | Train classifier needed for conditional-accuracy evaluation |
| `evaluate.py` | Score generated labels and visualize generated samples |
| `app.py` | Streamlit letter selection, generation, and PNG downloads |
| `ARCHITECTURE.md` | Architecture diagram and model explanation |
| `REPORT.md` | Project report template/content |
| `VIVA_QUESTIONS.md` | Example viva questions and answers |

## Requirements

- Python 3.10 or newer
- Internet access the first time, to download EMNIST
- CPU works; an NVIDIA GPU with a compatible PyTorch build is recommended for
  faster training.

The first install on Windows may take time because PyTorch is large. For a
GPU-specific PyTorch build, use the install command recommended by the
[official PyTorch selector](https://pytorch.org/get-started/locally/) and
then install the remaining packages from `requirements.txt`.

## Run locally on Windows

Open PowerShell in this project folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents venv activation, run this once for the current user or
use the venv interpreter directly:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 1. Inspect and visualize the dataset

```powershell
python visualize_data.py
```

The first run downloads EMNIST and saves `outputs/real_samples.png`. To also
open a plot window, add `--show`.

### 2. Train the cGAN

```powershell
python train.py --epochs 20
```

This downloads EMNIST Letters if needed and writes `checkpoints/cgan_latest.pt`,
periodic epoch checkpoints, generated-image grids under `outputs/`, and
`outputs/training_losses.png`. A short run can test the pipeline:

```powershell
python train.py --epochs 1 --batch-size 64
```

One epoch is only a smoke test, not a guarantee of convincing handwriting.
Training duration depends on hardware; inspect the actual losses and samples.

### 3. Train the evaluation classifier

```powershell
python train_classifier.py --epochs 5
```

This writes `checkpoints/classifier.pt` when held-out test accuracy improves.
The displayed classifier test accuracy is measured on real EMNIST test data;
the generated-image metric is a proxy, not a human quality score.

### 4. Evaluate and inspect generation

```powershell
python evaluate.py --samples-per-class 100
```

It reports the conditional-accuracy proxy over generated examples and saves
`outputs/evaluation_samples.png`. Both GAN and classifier checkpoints are
required.

### 5. Launch the interactive app

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Select a letter, choose the sample
count and random seed, generate, and download each PNG. The seed can reproduce
the same samples with the same model and software environment.

### 6. Run the offline smoke tests

```powershell
python -m unittest -v
```

## Run in Google Colab

1. Upload this project folder to Google Drive, or upload its source files to a
   Colab session.
2. In **Runtime → Change runtime type**, select a GPU if available.
3. Mount Drive and change to the project folder (adjust the folder name):

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/handwritten-cgan
   ```

4. Install dependencies if the runtime does not already have them:

   ```python
   %pip install -r requirements.txt
   ```

5. Run dataset visualization, training, classifier, and evaluation:

   ```python
   !python visualize_data.py
   !python train.py --epochs 20
   !python train_classifier.py --epochs 5
   !python evaluate.py --samples-per-class 100
   ```

6. Start the interface in Colab if desired. A tunnel or Colab-supported
   port-forwarding method may be needed to access Streamlit; a simpler option
   is running the app locally after copying the checkpoints from Drive.

   ```python
   !python -m streamlit run app.py --server.port 8501
   ```

   Do not expose a public tunnel unless you understand its access and privacy
   implications. Save the `checkpoints/` folder to persistent Drive storage;
   Colab's temporary local storage can disappear when the runtime ends.

## Model, data, and evaluation notes

- EMNIST Letters has 26 classes; the dataset's labels 1-26 are remapped to
  model indices 0-25.
- Images are oriented for reading and normalized from [0, 1] to [-1, 1].
- Generator output uses `tanh`; discriminator output is an un-sigmoided logit
  consumed by `BCEWithLogitsLoss`.
- The classifier is trained separately using only the training split and
  checked on the EMNIST test split. Its accuracy on generated samples is an
  imperfect recognition proxy, not a measure of artistic quality, diversity,
  or memorization.
- Checkpoints, downloaded data, and plots are excluded from version control.

## Troubleshooting

- **`No module named ...`:** activate `.venv` and run
  `python -m pip install -r requirements.txt` using that same interpreter.
- **PyTorch/torchvision import or operator error:** install a matching pair
  from the official PyTorch selector; avoid mixing incompatible CPU and CUDA
  wheels.
- **Download failure:** check internet access and retry. EMNIST files are
  cached in `data/` after a successful download.
- **Out of memory:** reduce batch size, for example
  `python train.py --batch-size 64`; lower it further if needed.
- **App says checkpoint missing:** complete GAN training from the project
  directory first. To show classifier predictions, run
  `python train_classifier.py` too.
- **Generated samples look poor:** GAN training can be unstable. Train for
  more epochs, view loss/sample grids, and compare the results; loss values
  alone do not guarantee good samples.
- **Streamlit command not found:** run `python -m streamlit run app.py` after
  installing requirements.
- **No GPU:** training falls back to CPU and may take substantially longer.

## Credits

Dataset: [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset).
Framework: [PyTorch](https://pytorch.org/).
