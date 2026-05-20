# TruthSight: Enhancement-Guided DeepFake Detection in Visual Media

A two-stage deepfake detection system that combines Retinex-guided U-Net image enhancement with ResNet50-based classification, deployed via a Flask web application.

---

## Features
- Upload any video and get Real/Fake prediction instantly
- Low-light image enhancement before classification
- Confidence score displayed with every prediction
- Downloadable analysis report
- Handles edge cases like no face detected

---

## Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV
- **Face Detection:** dlib, face_recognition
- **Enhancement Model:** Retinex-guided U-Net
- **Classification Model:** ResNet50 (Transfer Learning)
- **Web Framework:** Flask

---

## How It Works
1. User uploads a video
2. OpenCV extracts 1 frame/sec
3. dlib detects and crops face regions
4. U-Net enhances low-light frames
5. ResNet50 classifies each face as Real/Fake
6. Majority voting gives final prediction
7. Result displayed with confidence score

---

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 99.56% |
| Precision | 96.84% |
| Recall | 97.65% |
| F1-Score | 97.25% |
| U-Net MSE | 0.0566 |
| U-Net MAE | 0.2092 |

---

## Datasets
- **DFDC** (DeepFake Detection Challenge) — for classifier training
- **LOL** (Low-Light dataset) — for enhancement model training

---

## Installation

```bash
git clone https://github.com/yourusername/truthsight.git
cd truthsight
pip install -r requirements.txt
python app.py
```

---

## Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- dlib
- face_recognition
- Flask
- NumPy
- Pandas

---

## Results
- Without Enhancement: 97.3% accuracy
- With Enhancement: 99.56% accuracy
- Outperforms traditional CNN (93%) and XceptionNet (97%)

---

## Limitations
- Requires clearly visible faces in video
- No real-time stream support yet
- Performance may vary on unseen deepfake techniques

---

## Future Scope
- Real-time video stream detection
- Transformer-based models
- Multimodal audio + video analysis
- Mobile and cloud deployment
- Grad-CAM explainability

---

## Authors
- Ardra Padmakumar (LMC22CS036)
- Arsha J S (LMC22CS040)
- Arsha P A (LMC22CS041)
- Aswathy S (LMC22CS042)

Department of Computer Science and Engineering
Lourdes Matha College of Science and Technology, Trivandrum
March 2026
