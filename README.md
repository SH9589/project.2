# Emotion Detection System

A comprehensive AI-powered emotion detection system that analyzes emotions from facial expressions, voice tone, and text input using computer vision, speech processing, and natural language processing (NLP).

## Features

- **Facial Emotion Recognition**: Uses OpenCV and deep learning models to detect emotions from facial expressions
- **Voice-Based Emotion Detection**: Analyzes tone, pitch, and speech patterns using Librosa and ML models
- **Text Sentiment Analysis**: Implements NLP techniques to determine emotional tone of text input
- **Graphical User Interface (GUI)**: Interactive platform for user input and analysis
- **Multi-Modal Emotion Analysis**: Combines face, voice, and text detection for improved accuracy

## Requirements

- Python 3.8+
- OpenCV
- Librosa
- TensorFlow/Keras
- PyQt5
- NumPy
- Pandas
- scikit-learn
- Transformers (for BERT)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. Use the GUI to:
   - Upload images for facial emotion detection
   - Record voice for emotion analysis
   - Input text for sentiment analysis

## Project Structure

```
emotion-detection/
├── main.py              # Main application file
├── models/              # Trained models
├── utils/               # Utility functions
├── gui/                 # GUI components
├── data/                # Sample data
└── requirements.txt     # Project dependencies
```

## Applications

- Mental health monitoring
- AI chatbots
- Virtual assistants
- Customer sentiment analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details. 