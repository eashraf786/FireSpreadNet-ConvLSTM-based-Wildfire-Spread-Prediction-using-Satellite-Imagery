# FireSpreadNet-ConvLSTM-based-Wildfire-Spread-Prediction-using-Satellite-Imagery

This project implements a deep learning-based approach to predict wildfire spread using satellite imagery and environmental data. The model uses a ConvLSTM architecture to capture both spatial and temporal patterns in wildfire progression.

## Overview

The project aims to predict wildfire spread patterns using a sequence of satellite imagery and environmental data. The model takes into account various factors such as vegetation, terrain, weather conditions, and historical fire spread patterns to make accurate predictions.

## Features

- ConvLSTM-based deep learning model for spatiotemporal prediction
- Custom data loader for handling wildfire event sequences
- Comprehensive evaluation metrics (IoU, F1-Score, Precision, Recall, etc.)
- Adaptive early stopping mechanism
- Support for multiple input features and temporal sequences

## Project Structure

```
WildfireSpreadPrediction/
├── main.py                     # Main training script
├── preprocess_and_save_data.py # Data preprocessing utilities
├── DataLoader.py               # Custom dataset loader
├── buildmodel.py               # Model architecture definition
├── utils.py                    # Utility functions and custom metrics
├── savetifdata.py              # Save .tif files from WildfireSpreadTS dataset
├── eval_model.py               # Model evaluation and sample predictions
└── requirements.txt            # Project dependencies
```

## Requirements

The project requires the following dependencies:
- tensorflow>=2.8.0
- tensorflow-addons>=0.16.0
- numpy>=1.20.0
- matplotlib>=3.5.0
- scikit-image>=0.19.0
- scikit-learn>=1.0.0
- tqdm>=4.61.0
- opencv-python>=4.5.0
- rasterio>=1.2.0
- remotezip>=0.10.0
- pandas>=1.3.0
- jupyter>=1.0.0
- notebook>=6.4.0

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Data

The project uses the WildfireSpreadTS dataset, which contains wildfire event data organized by years (2018-2021). Each event contains:
- Satellite imagery sequences (multi-temporal)
- Environmental data (22 bands including vegetation indices, terrain features, and weather parameters)
- Fire spread masks (binary masks indicating burned areas)

## Model Architecture

The model uses a ConvLSTM architecture with the following components:
- Multiple ConvLSTM layers for spatiotemporal feature extraction
- Batch normalization and dropout for regularization
- Final convolutional layer for prediction
- Custom loss functions (Dice Loss, IoU Loss)
- Multiple evaluation metrics (IoU, F1-Score, Precision, Recall)

## Training

To train the model:
1. Prepare your data in the required format
2. Configure the parameters in `main.py`
3. Run the training script:
```bash
python main.py
```

The model uses adaptive early stopping and learning rate reduction to optimize training. Model checkpoints are saved whenever on every new best performing epoch.

## Evaluation

The model is evaluated using multiple metrics:
- Accuracy
- Dice Loss
- Intersection over Union (IoU)
- F1-Score
- Precision
- Recall
- Specificity
- Structural Similarity Index (SSIM)
- Mean Average Precision (mAP)
