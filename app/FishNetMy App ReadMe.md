# FishNetMy

FishNetMy is a desktop app for running object detection models on images, video, live camera feeds, and whole folders of photos. It's built with Tkinter for the interface and supports three model types: YOLOv11, Faster R-CNN, and Mask R-CNN. It runs on GPU when CUDA is available and falls back to CPU automatically.

## What it does

The app has four main modes, accessible from the top navigation bar:

**Live Inference**
Scans for connected cameras (including virtual cameras like Camo Studio or OBS Virtual Camera), lets you pick one, and runs detection on the live feed in real time. You can adjust the confidence threshold with a slider while it's running.

**Image Inference**
Upload a single image, run detection on it, then save the result or open it in your system's default photo viewer.

**Folder Inference**
Upload a folder of images and run detection on all of them at once. Results display in a paginated grid (30 images per page), and you can save every processed image in one batch.

**Video Inference**
Upload a video file and run detection frame by frame. Once processing finishes, the app switches to a playback mode with play/pause, a seek bar, and 10-second skip buttons. You can export the annotated video as an MP4.

## Model management

Models live in a `models/` folder next to the script, or you can upload your own `.pt` (YOLOv11) or `.pth` (Faster R-CNN / Mask R-CNN) file through the **Upload Model** button. Uploaded models get copied into an app-data folder so they persist between sessions:

- Windows: `%APPDATA%\FishNetMy\models`
- macOS/Linux: `~/.fishnetmy/models`

The app guesses which architecture a model uses based on its file extension and filename (for example, a `.pth` file with "mask" in the name loads as Mask R-CNN). For ambiguous `.pth` files, it inspects the checkpoint's internal keys to decide.

Right-click a model in the dropdown to:
- **Configure Classes** – set custom names and colors for each class the model detects, so bounding boxes show readable labels instead of raw class IDs.
- **Remove Model** – delete the model file and its config permanently.

There's also a **Configure** button in the nav bar that opens a manager dialog for handling all your models in one place.

The app remembers the last model you had loaded and reloads it automatically on startup.

## Requirements

Install these Python packages:

```bash
pip install opencv-python numpy torch torchvision pillow ultralytics
```

You'll also need Tkinter, which usually ships with Python. On Linux, if it's missing, install it with:

```bash
sudo apt-get install python3-tk
```

For GPU acceleration, install a CUDA-enabled build of PyTorch that matches your GPU driver. Check the [PyTorch install page](https://pytorch.org/get-started/locally/) for the right command. Without CUDA, the app still runs, just on CPU, and it will show a red "CPU only" indicator in the top-right corner instead of a green GPU label.

## How to run it

1. Place the script (`app_v5.py`) in a folder, and optionally add a `models/` subfolder with your `.pt` or `.pth` model files.
2. Run:

```bash
python app_v5.py
```

3. Pick a model from the **Model** dropdown in the nav bar, or click **Upload Model** to add one from disk.
4. Choose a mode (Live, Image, Folder, or Video) and start detecting.

## Notes

- Confidence threshold sliders (0.0 to 1.0) control how strict detection is. Lower values catch more objects but increase false positives.
- Saved images and videos keep the original filename with an `_inferred` suffix or a save dialog prompt, depending on the mode.
- Class configuration (names and colors) is saved per model, so switching models automatically applies the right labels.
- [Demo Video](https://youtu.be/SUBtskzbZbc)
