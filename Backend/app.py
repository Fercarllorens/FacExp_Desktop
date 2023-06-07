from flask import Flask, jsonify, request
import cv2
import csv
from deepface import DeepFace 
import time
import calendar
from datetime import timedelta
from threading import Thread
import json

# IMPORT OF HAARCASCADE FOR FACE DETECTION
face_cascade = cv2.CascadeClassifier('Backend//haarcascade//haarcascade_frontalface_default.xml')

# STATUS MEMORY
f = open("Backend//Status//config.json")
status_memory = json.load(f)
f.close()

# Logs stuff
timeInit = 0
timeIteration=0
id=0
headersCSV = ['id','steady_timestamp','uct_timestamp','HMS','dominant','neutral', 'happy', 'surprise','disgust', 'fear', 'sad', 'angry']
saveCSVFlag = False
reset_counter = 0

f = open('Backend//CSV//timestamptest.csv','w',newline='')
writer = csv.writer(f)
writer.writerow(headersCSV)
# VARIABLES FOR DEEPFACE EXPRESSIONS ~ SOFT CLASSIFIER ~
emotion_names = ['neutral', 'happy', 'surprise','disgust', 'fear', 'sad', 'angry']
emotion_confidence = [0, 0, 0, 0, 0, 0, 0]

#Variable camera
stream = cv2.VideoCapture(0)

#global variables analysis
txtEmotion = ""
angry = 0.0
disgust = 0.0
fear = 0.0
happy = 0.0
sad = 0.0
surprise = 0.0
neutral = 0.0


def start_analysis():
    global txtEmotion
    global angry
    global disgust
    global fear
    global happy
    global sad
    global surprise
    global neutral

    global stream
    global saveCSVFlag, reset_counter
    global timeInit
    global timeIteration
    global id

    while threadActive :
        ret, frame = stream.read()
        if ret == True:
            result = DeepFace.analyze(
                img_path=frame, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True)[0]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray,1.1,4)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
                                
            emotion = result['dominant_emotion']
            txtEmotion = str(emotion)

            angry = float(result['emotion']['angry'])
            disgust = float(result['emotion']['disgust'])
            fear = float(result['emotion']['fear'])
            happy = float(result['emotion']['happy'])
            sad = float(result['emotion']['sad'])
            surprise = float(result['emotion']['surprise'])
            neutral = float(result['emotion']['neutral'])

            global emotion_confidence
            emotion_confidence = [neutral, happy, surprise, disgust, fear, sad, angry]

            
            timeIteration = time.time()
            timeSteady = int(time.monotonic_ns() / 1000)
            timeuct = calendar.timegm(time.gmtime())
            timeSeconds = timeIteration - timeInit
            timeHHMMSS = timedelta(seconds=int(timeSeconds))
            writer.writerow([
                id,
                timeSteady,
                timeuct,
                timeHHMMSS,
                emotion,
                neutral,
                happy,
                surprise,
                disgust,
                fear,
                sad,
                angry
            ])
            id = id+1

            cv2.imwrite("Backend//Image//StreamRead.png", frame)
        else:
            stream.release()

#Thread analysis
threadActive = False

app = Flask(__name__)

@app.route('/result', methods= ['GET'])
def get_emotions():

    global txtEmotion
    global angry
    global disgust
    global fear
    global happy
    global sad
    global surprise
    global neutral

    if threadActive:
        return jsonify({"DominantEmotion":txtEmotion, 
                        "angry":str('%.4f'%(angry)).replace('.',','),
                        "disgust":str('%.4f'%(disgust)).replace('.',','),
                        "fear":str('%.4f'%(fear)).replace('.',','),
                        "happy":str('%.4f'%(happy)).replace('.',','),
                        "sad":str('%.4f'%(sad)).replace('.',','),
                        "surprise":str('%.4f'%(surprise)).replace('.',','),
                        "neutral":str('%.4f'%(neutral)).replace('.',',')
                        })
    else:
        return jsonify({"DominantEmotion":"neutral", 
                        "angry":0,
                        "disgust":0,
                        "fear":0,
                        "happy":0,
                        "sad":0,
                        "surprise":0,
                        "neutral":0
                        })

@app.route('/start', methods= ['GET'])
def startDF():
    global threadActive
    threadActive = True
    t = Thread(target=start_analysis, args=(), daemon=True)
    t.start()
    return jsonify({"Analysing": "TRUE"})

@app.route('/stop', methods= ['GET'])
def stopDF():
    global threadActive
    threadActive = False
    return jsonify({"Analysing": "FALSE"})

@app.route('/status', methods= ['GET'])
def sendStatus():
    return status_memory

@app.route('/models', methods=['POST'])
def updateModels():
    request_data = request.get_json()

    deepFaceNewStatus = request_data['DeepFace']
    transferNewStatus = request_data['Transfer']
    deepLearningNewStatus = request_data['DeepLearning']

    status_memory['Deepface'] = deepFaceNewStatus
    status_memory['Transfer Learning'] = transferNewStatus
    status_memory['Deep Learning'] = deepLearningNewStatus

    f = open("Backend//Status//config.json", 'w+')
    f.write(json.dumps(status_memory))
    f.close()

    return status_memory


if __name__ == "__main__":
    app.run(debug=True)