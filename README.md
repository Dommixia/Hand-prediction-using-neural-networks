# Hand Gesture Recognition using Neural Networks

A real-time hand gesture recognition system built using a Convolutional Neural Network (CNN) and OpenCV. The model detects and classifies hand gestures live from a webcam feed, trained on a custom augmented dataset.

## What this project does

The system captures hand gestures through a webcam in real time, processes the frames using OpenCV, and classifies the gesture using a trained CNN model. It was built from scratch — including collecting and augmenting the training dataset manually.

## How it works

1. **Data collection** — hand gesture images captured and stored in `dataset_aug/`
2. **Augmentation** — dataset augmented to increase variety and reduce overfitting
3. **Model training** — CNN trained using PyTorch, saved as `finger_model.pth`
4. **Real-time inference** — OpenCV captures webcam frames, model classifies each frame live

## Tech stack

- Python 3.10+
- PyTorch (CNN model)
- OpenCV (real-time webcam capture and image processing)
- NumPy

## Project structure
Hand-prediction-using-neural-networks/
│
├── OpenCV/              # Real-time inference scripts using webcam
├── dataset_aug/         # Augmented training dataset (dataset excluded due to privacy reasons)
├── finger_model.pth     # Trained CNN model weights
└── .gitignore

## Setup and usage

**1. Clone the repo**
```bash
git clone https://github.com/Dommixia/Hand-prediction-using-neural-networks.git
cd Hand-prediction-using-neural-networks
```

**2. Install dependencies**
```bash
pip install torch torchvision opencv-python numpy
```

**3. Run real-time inference**
```bash
cd OpenCV
python inference.py
```

## Key learnings

- How CNNs learn spatial features from image data
- The importance of data augmentation for small datasets
- How to connect a trained PyTorch model to a live OpenCV video feed
- Saving and loading model weights with `torch.save` and `torch.load`

## Future improvements

- Add more gesture classes
- Improve accuracy with transfer learning (e.g. MobileNet)
- Explore using MediaPipe for hand landmark detection

## Author

Built as a personal deep learning project exploring computer vision and neural networks.
