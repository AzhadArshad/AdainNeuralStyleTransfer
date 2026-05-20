# StyleForge AI — Neural Style Transfer

> **Trained from scratch.** No pretrained style transfer model was used — the decoder network was designed and entirely self trained , using our own training pipeline (`train.py`). Only the VGG19 encoder weights (used purely as a frozen feature extractor) are borrowed from a standard pretrained backbone.

Transform any photo into a work of art using deep learning. Upload a content image, pick a style reference, and StyleForge reconstructs your image with the textures, colors, and brushstrokes of the style — all in seconds.

---

## What Problem Does It Solve?

Traditional artistic style transfer required expensive optimization loops that took minutes per image. StyleForge uses **Adaptive Instance Normalization (AdaIN)** — a feed-forward approach that transfers style in a single forward pass. The result is near real-time stylization without sacrificing quality.

---

## Tech Stack

| Layer            | Technology                            |
| ---------------- | ------------------------------------- |
| Deep Learning    | PyTorch 2.2.2, torchvision 0.17.2     |
| Web Framework    | Flask 3.1.2, Flask-WTF                |
| Image Processing | Pillow 12.0.0                         |
| Frontend         | Bootstrap 5.3, Vanilla JS, Canvas API |
| Server           | Gunicorn                              |
| Deployment       | Render                                |

---

## Architecture

The model is split into two parts:

### Encoder — VGG19 (frozen)

A pre-trained VGG19 network truncated at `relu4_1`. It is never updated during training — it acts purely as a feature extractor. Features are extracted at four depths (`relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`) to capture both low-level textures and high-level structure.

### Decoder (trained from scratch)

A mirror of the VGG encoder using transposed convolutions and nearest-neighbor upsampling. It learns to reconstruct a realistic RGB image from the AdaIN-transformed feature space. This is the only part that is trained.

### AdaIN — The Core Idea

AdaIN normalizes the content image's feature statistics (mean and variance) to match those of the style image, channel by channel:

```
AdaIN(content, style) = style_std * ((content - mean(content)) / std(content)) + mean(style)
```

This single operation is what transfers the "feel" of the style image onto the structure of the content image, without any iterative optimization at inference time.

### Alpha Blending

An alpha parameter (0–1) lets users control stylization strength by linearly interpolating between the original content features and the AdaIN-transformed features before decoding:

```
features = α * AdaIN(content, style) + (1 - α) * content
```

---

## Training

### Loss Function

Training optimizes a weighted combination of two losses:

- **Content Loss** — MSE between the decoder output's features and the AdaIN target features. Ensures the decoded image preserves content structure.
- **Style Loss** — MSE between the channel-wise mean and standard deviation of the decoder output and the style image, computed at all four VGG encoder levels. Ensures the decoded image picks up the style's textures.

```
Total Loss = content_loss + λ * style_loss
```

### Optimizer

Adam optimizer with a lambda learning rate scheduler that decays over training.

### Experiments

Multiple training runs were conducted before arriving at the final model:

| Run           | Notes                                                                     |
| ------------- | ------------------------------------------------------------------------- |
| `trial`       | Initial proof-of-concept run, high style weight                           |
| `experiment2` | Tuned content/style loss balance                                          |
| `m2_fast`     | Reduced image resolution for faster iteration                             |
| `m2_fast2`    | Further speed experiments with smaller batches                            |
| `final`       | Best checkpoint — `decoder_5.pth` selected based on visual output quality |

The key finding across experiments: the style loss weight needs careful tuning — too high and the content structure collapses; too low and the style barely shows. The final model uses a balanced λ that preserves recognizable content while visibly transferring style.

---

## Project Structure

```
nst/
├── app.py                  # Flask app — routes, inference pipeline
├── train.py                # Training script
├── decoder.pth             # Trained decoder weights
├── vgg_normalised.pth      # Pre-trained VGG19 encoder weights
├── utils/
│   ├── models.py           # VGGEncoder + Decoder architecture
│   └── utils.py            # AdaIN op, dataset, transforms
├── templates/index.html    # Web UI
├── static/uploads/         # User-uploaded and output images
└── examples/               # Demo images shown on the homepage
```

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the dev server
python app.py
```

Open `http://localhost:5000`, upload a content and style image, adjust the alpha slider, and hit **Transfer Style**.

---

## Deploying on Render

1. Push the repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn --bind :$PORT app:app`
5. Add `SECRET_KEY` as an environment variable in Render's dashboard

---

## Results

The model handles a wide range of styles — pencil sketches, Picasso-style cubism, impressionist paintings, watercolors, and more. Inference runs in under a second on CPU for 512×512 images.

---

## Learning Outcomes

- **Understood AdaIN from scratch** — went beyond using a pretrained model to actually implementing and training the full AdaIN pipeline, including the encoder-decoder architecture and the feature statistics alignment that makes style transfer work.

- **Training intuition** — learned how sensitive deep learning models are to loss weighting. Small changes to the content/style loss ratio produced drastically different visual results, building real intuition for hyperparameter tuning.

- **Experiment management** — ran multiple training runs with different configurations and learned to track, compare, and select checkpoints based on qualitative output rather than just loss curves.

- **End-to-end ML deployment** — took a model from a training script all the way to a live web app, handling everything in between: model serialization, Flask integration, static file serving, gunicorn, and cloud deployment on Render.

- **Full-stack thinking** — built both the ML backend and the frontend UI, understanding how model outputs flow through a web server to a user-facing interface.

- **Debugging across the stack** — encountered and fixed issues at every layer: PyTorch device handling (CUDA/MPS/CPU), Flask form validation, CSRF tokens, file upload handling, and frontend image previewing.

---

## Acknowledgements

A huge thanks to **Apna College** for their teachings and for building a community that makes learning deep learning and software development accessible. This project would not have been possible without the foundation they helped build.

---

_Built with PyTorch + Flask · Deployed on Render_
