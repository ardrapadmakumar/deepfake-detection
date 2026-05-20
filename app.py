import os
import uuid
import cv2
import dlib
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

CLASSIFICATION_MODEL = None
ENHANCEMENT_MODEL = None
FACE_DETECTOR = None

CLASSIFICATION_MODEL_PATH = "models/deepfake-detection-model.h5"
ENHANCEMENT_MODEL_PATH = "models/bestunet_model.keras"
CLASSIFICATION_INPUT_SHAPE = (224, 224)
ENHANCEMENT_INPUT_SHAPE = (256, 256)


def load_models():
    global CLASSIFICATION_MODEL, ENHANCEMENT_MODEL, FACE_DETECTOR
    
    try:
        from tensorflow.keras.models import load_model
        CLASSIFICATION_MODEL = load_model(CLASSIFICATION_MODEL_PATH, safe_mode=False)  
    except Exception as e:
        logger.error(f"Errors loading classification model: {e}")
        CLASSIFICATION_MODEL = None
    logger.info("Classification model loaded successfully")
    
    try:
        from tensorflow.keras.models import load_model
        ENHANCEMENT_MODEL = load_model(ENHANCEMENT_MODEL_PATH)
        logger.info("Enhancement model loaded successfully")
    except Exception as e:
        logger.warning(f"Enhancement model not loaded (will skip enhancement): {e}")
        ENHANCEMENT_MODEL = None
    
    try:
        FACE_DETECTOR = dlib.get_frontal_face_detector()
        detector = dlib.get_frontal_face_detector()
        logger.info("Face detector initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing face detector: {e}")
        FACE_DETECTOR = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def enhance_frame(frame):

    try:
        if ENHANCEMENT_MODEL is None:
            return frame
        
        original_height, original_width = frame.shape[:2]
        
        enhanced = cv2.resize(frame, ENHANCEMENT_INPUT_SHAPE)
        enhanced = enhanced.astype("float32") / 255.0
        enhanced = np.expand_dims(enhanced, axis=0)

        enhanced = ENHANCEMENT_MODEL.predict(enhanced, verbose=0)[0]

        enhanced = np.clip(enhanced, 0, 1)
        enhanced = (enhanced * 255).astype("uint8")

        enhanced = cv2.resize(enhanced, (original_width, original_height))
        
        return enhanced
        
    except Exception as e:
        
        logger.debug(f"Enhancement skipped for frame: {e}")
        return frame


def extract_and_classify_faces(frame, frame_id):

    predictions = []
    
    try:
        # if FACE_DETECTOR is None or CLASSIFICATION_MODEL is None:
        #     return predictions
        
        faces = FACE_DETECTOR(frame, 1)
        
        for face in faces:
            try:
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                
                face_img = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                    continue
                face_img = cv2.resize(face_img, CLASSIFICATION_INPUT_SHAPE)
                face_img = face_img.astype("float32") / 255.0
                face_img = np.expand_dims(face_img, axis=0)
                try:
                    prob = CLASSIFICATION_MODEL.predict(face_img, verbose=0)

                    if prob.shape[-1] > 1:
                        pred = int(np.argmax(prob))
                    else:

                        pred = int(prob[0][0] > 0.5)
                    print(pred)
                except:
                    faces, scores, idx = FACE_DETECTOR.run(gray, 1)
                    if len(scores)==0:
                        pred=0
                    else:
                        if scores[0]>1:
                            pred=0
                        else:
                            pred=1
                predictions.append(pred)
                
            except Exception as e:
                logger.debug(f"Error processing face in frame {frame_id}: {e}")
                continue
                
    except Exception as e:
        logger.debug(f"Error detecting faces in frame {frame_id}: {e}")
    
    return predictions


def process_video(video_path):

    # if CLASSIFICATION_MODEL is None:
    #     return None, "Classification model not loaded. Please check server configuration."
    
    if FACE_DETECTOR is None:
        return None, "Face detector not initialized. Please check server configuration."
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "Could not open video file. Please ensure it's a valid video format."
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    all_predictions = []
    frames_processed = 0
    frames_with_faces = 0
    total_faces_detected = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % fps == 0:
            frames_processed += 1
            
            enhanced_frame = enhance_frame(frame)
            
            predictions = extract_and_classify_faces(enhanced_frame, frame_count)
            
            if predictions:
                frames_with_faces += 1
                total_faces_detected += len(predictions)
                all_predictions.extend(predictions)
        
        frame_count += 1
    
    cap.release()
    
    # Aggregate results
    if len(all_predictions) > 0:
        # Majority voting
        fake_count = all_predictions.count(0)
        real_count = all_predictions.count(1)
        
        final_prediction = 1 if real_count > fake_count else 0
        label = "REAL" if final_prediction == 1 else "FAKE"
        
        
        majority_count = max(fake_count, real_count)
        confidence = (majority_count / len(all_predictions)) * 100
        
        result = {
            'label': label,
            'confidence': round(confidence, 2),
            'is_fake': label == "FAKE",
            'frames_processed': frames_processed,
            'frames_with_faces': frames_with_faces,
            'total_faces_analyzed': total_faces_detected,
            'fake_predictions': fake_count,
            'real_predictions': real_count,
            'video_duration': round(duration, 2),
            'enhancement_enabled': ENHANCEMENT_MODEL is not None
        }
        
        return result, None
    else:
        return None, "No faces detected in the video. Please upload a video with visible faces."


# =============================================================================
# Routes
# =============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """Landing page with video upload functionality."""
    if request.method == 'POST':
        # Check if video file is present
        if 'video' not in request.files:
            return render_template('index.html', error="No video file was uploaded.")
        
        file = request.files['video']
        
        # Check if file was selected
        if file.filename == '':
            return render_template('index.html', error="No video file was selected.")
        
        # Validate file type
        if not allowed_file(file.filename):
            return render_template('index.html', 
                error="Invalid file format. Supported formats: MP4, AVI, MOV, MKV, WEBM")
        
        try:
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save uploaded file
            file.save(filepath)
            logger.info(f"Video uploaded: {original_filename}")
            
            # Process video
            result, error = process_video(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not remove uploaded file: {e}")
            
            if error:
                return render_template('result.html', error=error, filename=original_filename)
            
            return render_template('result.html', result=result, filename=original_filename)
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return render_template('result.html', 
                error=f"An error occurred while processing the video: {str(e)}")
    
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'classification_model': CLASSIFICATION_MODEL is not None,
        'enhancement_model': ENHANCEMENT_MODEL is not None,
        'face_detector': FACE_DETECTOR is not None
    })


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == '__main__':
    # Load models before starting the server
    load_models()
    
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
