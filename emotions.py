import cv2
import numpy as np
import dlib
from imutils import face_utils

from statistics import mode
from datasets import get_labels
from inference import detect_faces
from inference import draw_text
from inference import draw_bounding_box
from inference import apply_offsets
from inference import load_detection_model
from preprocessor import preprocess_input
from tensorflow.keras.models import load_model
USE_WEBCAM = True 


emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
model_path = "E:/Face-and-Emotion-Recognition-master/letstrygender.h5"
model = load_model(model_path)
#model_path1 = "E:/Face-and-Emotion-Recognition-master/letstryage.h5"
model_path1 = "E:/cnn_age_gender-main/letstryage_3.h5"
model1 = load_model(model_path1)
gender_ = []
age_=[]

m = "E:/Face-and-Emotion-Recognition-master/gender_CNN.h5"
m1 = load_model(m)
m2 = []
labels =["0-3",  
        "4-6",      
        "7-10",
        "11-20",
        "21-30", 
        "31-40",        
        "41-60",
        "61-70",
        "71-80",
        "81-90",
        "90+",
        ]
'''labels =["0-2",  
        "3-6",      
        "7-12",
        "13-17",
        "18-24", 
        "25-35",        
        "36-44",
        "45-52",
        "53-60",
        "61-70",
        "71-80",
        "81-95",
        "95+",
        ]'''
labels1 = ["Male" , "Female"]

frame_window = 10
emotion_offsets = (20, 40)


detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)

emotion_target_size = emotion_classifier.input_shape[1:3]

emotion_window = []
gender_=[]


cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)


cap = None
USE_WEBCAM = True
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) 
else:
    cap = cv2.VideoCapture('./test/testvdo.mp4') 

while cap.isOpened(): 
    ret, bgr_image = cap.read()

    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = detector(rgb_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        img=gray_face
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        try:
            img = cv2.resize(img,(200,200))
        except:
            continue
        try:
            c = cv2.resize(img,(32,32))
        except:
            continue
        predict2=model1.predict(c.reshape(-1,32,32,3))
        age_.append(predict2)
        
        predict_index = np.argmax(predict2)
        a=labels[predict_index]
        
        p3=model.predict(c.reshape(-1,32,32,3))
        m2.append(p3)
        g = np.argmax(p3)
        k = labels1[g]
        
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text +" "+" "+a +" "+k)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
