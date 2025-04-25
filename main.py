import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                           QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import librosa
import tensorflow as tf
from transformers import pipeline

class EmotionDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Detection System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.init_ui()
        self.init_models()
        
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tabs for different emotion detection modes
        self.create_face_detection_tab()
        self.create_voice_detection_tab()
        self.create_text_detection_tab()
        
        # Add tabs to main layout
        layout.addWidget(self.face_tab)
        layout.addWidget(self.voice_tab)
        layout.addWidget(self.text_tab)
        
    def create_face_detection_tab(self):
        self.face_tab = QWidget()
        face_layout = QVBoxLayout(self.face_tab)
        
        # Camera feed display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        face_layout.addWidget(self.camera_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_camera_btn = QPushButton("Start Camera")
        self.stop_camera_btn = QPushButton("Stop Camera")
        self.upload_image_btn = QPushButton("Upload Image")
        
        button_layout.addWidget(self.start_camera_btn)
        button_layout.addWidget(self.stop_camera_btn)
        button_layout.addWidget(self.upload_image_btn)
        face_layout.addLayout(button_layout)
        
        # Connect signals
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.upload_image_btn.clicked.connect(self.upload_image)
        
    def create_voice_detection_tab(self):
        self.voice_tab = QWidget()
        voice_layout = QVBoxLayout(self.voice_tab)
        
        # Voice recording controls
        self.record_btn = QPushButton("Record Voice")
        self.stop_record_btn = QPushButton("Stop Recording")
        self.analyze_voice_btn = QPushButton("Analyze Voice")
        
        voice_layout.addWidget(self.record_btn)
        voice_layout.addWidget(self.stop_record_btn)
        voice_layout.addWidget(self.analyze_voice_btn)
        
        # Connect signals
        self.record_btn.clicked.connect(self.start_recording)
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.analyze_voice_btn.clicked.connect(self.analyze_voice)
        
    def create_text_detection_tab(self):
        self.text_tab = QWidget()
        text_layout = QVBoxLayout(self.text_tab)
        
        # Text input
        self.text_input = QTextEdit()
        self.analyze_text_btn = QPushButton("Analyze Text")
        
        text_layout.addWidget(self.text_input)
        text_layout.addWidget(self.analyze_text_btn)
        
        # Connect signals
        self.analyze_text_btn.clicked.connect(self.analyze_text)
        
    def init_models(self):
        # Initialize face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize text sentiment analysis model
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Initialize voice emotion detection model
        # (This would be replaced with your actual model)
        self.voice_model = None
        
    def start_camera(self):
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms
        
    def stop_camera(self):
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'camera'):
            self.camera.release()
            
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and emotions
            faces = self.face_cascade.detectMultiScale(rgb_frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Convert to QImage and display
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
            
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            image = cv2.imread(file_name)
            # Process image and detect emotions
            # (Implement your image processing logic here)
            
    def start_recording(self):
        # Implement voice recording logic
        pass
        
    def stop_recording(self):
        # Implement stop recording logic
        pass
        
    def analyze_voice(self):
        # Implement voice analysis logic
        pass
        
    def analyze_text(self):
        text = self.text_input.toPlainText()
        if text:
            result = self.sentiment_analyzer(text)
            QMessageBox.information(self, "Analysis Result", f"Emotion: {result[0]['label']}\nScore: {result[0]['score']}")
        else:
            QMessageBox.warning(self, "Warning", "Please enter some text to analyze")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionDetectionApp()
    window.show()
    sys.exit(app.exec_()) 