from flask import Flask, jsonify, request
import cv2
import csv
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepface import DeepFace 
import time
import calendar
from datetime import timedelta
from threading import Thread
import json
import numpy as np
from deep_emotion import Deep_Emotion
from mss import mss
from PIL import Image
import win32gui
from collections import Counter

# IMPORT OF HAARCASCADE FOR FACE DETECTION
face_cascade = cv2.CascadeClassifier('Backend//haarcascade//haarcascade_frontalface_default.xml')

#IMPORT MODELS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Deep_Emotion()
net.load_state_dict(torch.load('Backend//Models//deep_emotion-100-128-0.005.pt'))
net.to(device)

model = tf.keras.models.load_model('Backend//Models//Final_model_02')

# STATUS MEMORY
f = open("Backend//Status//config.json")
status_memory = json.load(f)
f.close()

# Logs stuff
timeInit = 0
timeIteration=0
id=0
headersCSV = ['id','HMS', 'task', 'dominantDF','dominantTL','dominantDL','neutralDF', 'happyDF', 'surpriseDF','disgustDF', 'fearDF', 'sadDF', 'angryDF','neutralTL', 'happyTL', 'surpriseTL','disgustTL', 'fearTL', 'sadTL', 'angryTL','neutralDL', 'happyDL', 'surpriseDL','disgustDL', 'fearDL', 'sadDL', 'angryDL']
reset_counter = 0

# VARIABLES FOR DEEPFACE EXPRESSIONS ~ SOFT CLASSIFIER ~
emotion_names = ['neutral', 'happy', 'surprise','disgust', 'fear', 'sad', 'angry']
emotion_confidenceDF = [0, 0, 0, 0, 0, 0, 0]
emotion_confidenceTL = [0, 0, 0, 0, 0, 0, 0]
emotion_confidenceDL = [0, 0, 0, 0, 0, 0, 0]

#Variable camera
stream = cv2.VideoCapture(status_memory['Webcam'])

#global variables analysis
txtEmotionDF = ""
txtEmotionTL = ""
txtEmotionDL = ""

#Threshold analysis
windowDF = []
windowTL = []
windowDL = []

#Stream values
ret = None
frame = None
face_roi = None

#Task flag
task = 0;
lasttask = 0;

def start_stream():
    global stream
    global status_memory
    global ret
    global frame
    global face_roi

    while threadingActive:
        ret, frame = stream.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray,1.1,4)

            for (x,y,w,h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
                multiface = face_cascade.detectMultiScale(roi_gray)
                if len(multiface) == 0:
                    print('No faces')
                else:
                    for (ex, ey, ew, eh) in multiface:
                        face_roi = roi_color[ey:ey+eh, ex:ex+ew] 
        cv2.imwrite("Backend//Image//StreamRead.png", frame)
    
def start_webcam_record():
    global stream
    global status_memory

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20.0  # Controls the fps of the video created: todo look up optimal fps for webcam
    out = cv2.VideoWriter()
    if status_memory['Video Folder'] == 'Default':
        success = out.open('Backend//Video//output.mp4',fourcc, fps, (1920,1080),True)
    else: 
        success = out.open(status_memory['Video Folder'],fourcc, fps, (1920,1080),True)

    while threadingActive:
        ret, frame = stream.read()
        if ret == True:
            record = cv2.resize(frame, (1920,1080))
            out.write(record)
    out.release()

def start_screen_record():
    Xs = [0,8,6,14,12,4,2,0]
    Ys = [0,2,4,12,14,6,8,0] 

    sct = mss()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20.0  # Controls the fps of the video created: todo look up optimal fps for webcam
    out = cv2.VideoWriter()
    if status_memory['Screen Folder'] == 'Default':
        success = out.open('Backend//Screen//output.mp4',fourcc, fps, (1920,1080),True)
    else:
        success = out.open(status_memory['Screen Folder'], fourcc, fps, (1920,1080),True)

    while threadingActive:

        mouseX,mouseY = win32gui.GetCursorPos()
        if mouseX < 0:
            mouseX += 1920
        mouseX *= 1
        mouseY *= 1

        print(mouseX)
        

        sct_img = sct.grab(sct.monitors[status_memory['Screencam']])

        frame = np.array(sct_img)
        

        Xthis = [2*x+mouseX -10 for x in Xs]
        Ythis = [2*y+mouseY -40 for y in Ys]
        points = list(zip(Xthis,Ythis))
        points = np.array(points, 'int32')
        cv2.fillPoly(frame,[points],color=[255,255,255])
        image = cv2.resize(frame, (1920,1080))
        image = image[:,:,:3]
        out.write(image)
    out.release()

        

def start_analysis_deepFace():
    global txtEmotionDF
    global stream
    global status_memory
    global ret
    global frame
    while threadingActive and status_memory['DeepFace']:
        if ret == True:
            result = DeepFace.analyze(
                    img_path=frame, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    silent=True)[0]
            
            emotionDF = result['dominant_emotion']
            txtEmotionDF = str(emotionDF)

            angryDF = float(result['emotion']['angry'])
            disgustDF = float(result['emotion']['disgust'])
            fearDF = float(result['emotion']['fear'])
            happyDF = float(result['emotion']['happy'])
            sadDF = float(result['emotion']['sad'])
            surpriseDF = float(result['emotion']['surprise'])
            neutralDF = float(result['emotion']['neutral']) 

            global emotion_confidenceDF
            emotion_confidenceDF = [neutralDF, happyDF, surpriseDF, disgustDF, fearDF, sadDF, angryDF]
            
            global windowDF
            if status_memory['Threshold mode']:
                timeNow = time.time()
                instantAnalysis = {"time":timeNow, "analysis":emotion_confidenceDF}
                windowDF.append(instantAnalysis)
                
                timeThreshold = timeNow - (status_memory["Threshold"]/2)/1000

                for element in windowDF:
                    if element["time"] < timeThreshold:
                        windowDF.remove(element)
                    else:
                        break

                


def start_analysis_transfer():
    global txtEmotionTL

    global stream

    global status_memory

    global emotion_confidenceTL

    global ret
    global frame
    global face_roi
    
    while threadingActive and status_memory['Transfer Learning']:
        if ret == True: 

            if face_roi is not None:
                predictImageTL = cv2.resize(face_roi,(224,224))
                predictImageTL = np.expand_dims(predictImageTL,axis=0) 
                predictImageTL = predictImageTL/255.0 

                PredictionsTL = model.predict(predictImageTL)
            else:
                PredictionsTL = None

            if PredictionsTL is not None:
                angryTL = float(PredictionsTL[0][0])
                disgustTL = float(PredictionsTL[0][1])
                fearTL = float(PredictionsTL[0][2])
                happyTL = float(PredictionsTL[0][3])
                neutralTL = float(PredictionsTL[0][4])
                sadTL = float(PredictionsTL[0][5])
                surpriseTL = float(PredictionsTL[0][6])
            
                emotion_confidenceTL = [neutralTL*100, happyTL*100, surpriseTL*100, disgustTL*100, fearTL*100, sadTL*100, angryTL*100]

            if PredictionsTL is not None:
                txtEmotionTL = emotion_names[np.argmax(emotion_confidenceTL)]
            else:
                txtEmotionTL = ""
            
            global windowTL
            if status_memory['Threshold mode']:
                timeNow = time.time()
                instantAnalysis = {"time":timeNow, "analysis":emotion_confidenceTL}
                windowTL.append(instantAnalysis)
                
                timeThreshold = timeNow - (status_memory["Threshold"]/2)/1000

                for element in windowTL:
                    if element["time"] < timeThreshold:
                        windowTL.remove(element)
                    else:
                        break

def start_analysis_deepLearning():
    global txtEmotionDL

    global stream

    global status_memory

    global emotion_confidenceDL

    global ret
    global frame
    global face_roi
    
    while threadingActive and status_memory['Deep Learning']:
        if ret == True:

            if face_roi is not None:
                graytemp = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                predictImageDL = cv2.resize(graytemp,(48,48))
                predictImageDL = np.expand_dims(predictImageDL,axis=0)
                predictImageDL = np.expand_dims(predictImageDL,axis=0)
                predictImageDL = predictImageDL/255.0

                data = torch.from_numpy(predictImageDL)
                data = data.type(torch.FloatTensor)
                data = data.to(device)
                output = net(data)

                PredictionsDL = F.softmax(output, dim=1)
            else:
                PredictionsDL = None

            if PredictionsDL is not None:
                angryDL = float(PredictionsDL[0][0])
                disgustDL = float(PredictionsDL[0][1])
                fearDL = float(PredictionsDL[0][2])
                happyDL = float(PredictionsDL[0][3])
                sadDL = float(PredictionsDL[0][4])
                surpriseDL = float(PredictionsDL[0][5])
                neutralDL = float(PredictionsDL[0][6])
            
                emotion_confidenceDL = [neutralDL*100, happyDL*100, surpriseDL*100, disgustDL*100, fearDL*100, sadDL*100, angryDL*100]
            
            if PredictionsDL is not None:
                txtEmotionDL = emotion_names[np.argmax(emotion_confidenceDL)]
            else:
                txtEmotionDL = ""

            global windowDL
            if status_memory['Threshold mode']:
                timeNow = time.time()
                instantAnalysis = {"time":timeNow, "analysis":emotion_confidenceDL}
                windowDL.append(instantAnalysis)
                
                timeThreshold = timeNow - (status_memory["Threshold"]/2)/1000

                for element in windowDL:
                    if element["time"] < timeThreshold:
                        windowDL.remove(element)
                    else:
                        break

def start_log() :
    global txtEmotionDF
    global txtEmotionTL
    global txtEmotionDL

    global emotion_confidenceDF
    global emotion_confidenceTL
    global emotion_confidenceDL

    global stream
    global reset_counter
    global timeInit
    global timeIteration
    global id

    global status_memory
    global task

    if status_memory['Logs Folder'] == 'Default (folder: CSV)':
        f = open('Backend//CSV//timestamptest.csv','w',newline='')
        writer = csv.writer(f)
        writer.writerow(headersCSV)
    else:
        f = open(status_memory['Logs Folder'],'w',newline='')
        writer = csv.writer(f)
        writer.writerow(headersCSV)

    timeInit = time.time()
    time.sleep(3)
    while threadingActive and status_memory['Logs']:
        timeIteration = time.time()
        timeSeconds = timeIteration - timeInit
        timeHHMMSS = timedelta(seconds=int(timeSeconds))

        if status_memory['DeepFace']:
            neutralDF = emotion_confidenceDF[0]
            happyDF = emotion_confidenceDF[1]
            surpriseDF = emotion_confidenceDF[2]
            disgustDF = emotion_confidenceDF[3]
            fearDF = emotion_confidenceDF[4]
            sadDF = emotion_confidenceDF[5]
            angryDF = emotion_confidenceDF[6]
        else:
            neutralDF = 0
            happyDF = 0
            surpriseDF = 0
            disgustDF = 0
            fearDF = 0
            sadDF = 0
            angryDF = 0

        if status_memory['Transfer Learning']:
            neutralTL = emotion_confidenceTL[0]
            happyTL = emotion_confidenceTL[1]
            surpriseTL = emotion_confidenceTL[2]
            disgustTL = emotion_confidenceTL[3]
            fearTL = emotion_confidenceTL[4]
            sadTL = emotion_confidenceTL[5]
            angryTL = emotion_confidenceTL[6]
        else:
            neutralTL = 0
            happyTL = 0
            surpriseTL = 0
            disgustTL = 0
            fearTL = 0
            sadTL = 0
            angryTL = 0

        if status_memory['Deep Learning']:
            neutralDL = emotion_confidenceDL[0]
            happyDL = emotion_confidenceDL[1]
            surpriseDL = emotion_confidenceDL[2]
            disgustDL = emotion_confidenceDL[3]
            fearDL = emotion_confidenceDL[4]
            sadDL = emotion_confidenceDL[5]
            angryDL = emotion_confidenceDL[6]
        else:
            neutralDL = 0
            happyDL = 0
            surpriseDL = 0
            disgustDL = 0
            fearDL = 0
            sadDL = 0
            angryDL = 0

        writer.writerow([
            id,
            timeHHMMSS,
            task,
            txtEmotionDF,
            txtEmotionTL,
            txtEmotionDL,
            str('%.4f'%(neutralDF)).replace('.',','),
            str('%.4f'%(happyDF)).replace('.',','),
            str('%.4f'%(surpriseDF)).replace('.',','),
            str('%.4f'%(disgustDF)).replace('.',','),
            str('%.4f'%(fearDF)).replace('.',','),
            str('%.4f'%(sadDF)).replace('.',','),
            str('%.4f'%(angryDF)).replace('.',','),
            str('%.4f'%(neutralTL)).replace('.',','),
            str('%.4f'%(happyTL)).replace('.',','),
            str('%.4f'%(surpriseTL)).replace('.',','),
            str('%.4f'%(disgustTL)).replace('.',','),
            str('%.4f'%(fearTL)).replace('.',','),
            str('%.4f'%(sadTL)).replace('.',','),
            str('%.4f'%(angryTL)).replace('.',','),
            str('%.4f'%(neutralDL)).replace('.',','),
            str('%.4f'%(happyDL)).replace('.',','),
            str('%.4f'%(surpriseDL)).replace('.',','),
            str('%.4f'%(disgustDL)).replace('.',','),
            str('%.4f'%(fearDL)).replace('.',','),
            str('%.4f'%(sadDL)).replace('.',','),
            str('%.4f'%(angryDL)).replace('.',',')
        ])
        id = id+1
        time.sleep(0.1)
    f.close()
    if status_memory['Logs Folder'] == 'Default (folder: CSV)':
        f = open('Backend//CSV//timestamptest.csv')
        fout = open('Backend//CSV//timestamptest.txt','w',newline='')
    else:
        f = open(status_memory['Logs Folder'])
        name = status_memory['Logs Folder'][:-4]
        fout = open(name + '.txt','w',newline='')
    
    lines = f.readlines()

    actualTask = 0
    lastLineTask = 0
    arrayTaskEmotionsDF = []
    arrayTaskEmotionsTL = []
    arrayTaskEmotionsDL = []

    lines = lines[1:]

    if lines[1].split(',')[2] == 1:
        actualTask = 1
        lastLineTask = 1

    for line in lines:
        rowArray = line.split(',')
        if rowArray[2] != 0:
            actualTask = int(rowArray[2])
            if actualTask != lastLineTask and lastLineTask == 0:
                lastLineTask = actualTask
                arrayTaskEmotionsDF.append(rowArray[3])
                arrayTaskEmotionsTL.append(rowArray[4])
                arrayTaskEmotionsDL.append(rowArray[5])
            elif actualTask == lastLineTask:
                arrayTaskEmotionsDF.append(rowArray[3])
                arrayTaskEmotionsTL.append(rowArray[4])
                arrayTaskEmotionsDL.append(rowArray[5])
            elif actualTask != lastLineTask and lastLineTask != 0:
                fout.write('TASK ' + str(lastLineTask) + ' SUMARY :\n')
                fout.write('\n')
                if status_memory['DeepFace']:
                    dic = Counter(arrayTaskEmotionsDF)
                    dominant = max(dic, key=dic.get)
                    fout.write('Deepface sumary:\n')
                    fout.write('Dominant emotion in task: ' + dominant + '\n')
                    fout.write('Angry times: ' + str(dic['angry']) + '\n')
                    fout.write('Disgust times: ' + str(dic['disgust']) + '\n')
                    fout.write('Fear times: ' + str(dic['fear']) + '\n')
                    fout.write('Happy times: ' + str(dic['happy']) + '\n')
                    fout.write('Neutral times: ' + str(dic['neutral']) + '\n')
                    fout.write('Sad times: ' + str(dic['sad']) + '\n')
                    fout.write('Surprise times: ' + str(dic['surprise']) + '\n')
                    fout.write('\n')

                if status_memory['Transfer Learning']:
                    dic = Counter(arrayTaskEmotionsTL)
                    dominant = max(dic, key=dic.get)
                    fout.write('Transfer Learning sumary:\n')
                    fout.write('Dominant emotion in task: ' + dominant + '\n')
                    fout.write('Angry times: ' + str(dic['angry']) + '\n')
                    fout.write('Disgust times: ' + str(dic['disgust']) + '\n')
                    fout.write('Fear times: ' + str(dic['fear']) + '\n')
                    fout.write('Happy times: ' + str(dic['happy']) + '\n')
                    fout.write('Neutral times: ' + str(dic['neutral']) + '\n')
                    fout.write('Sad times: ' + str(dic['sad']) + '\n')
                    fout.write('Surprise times: ' + str(dic['surprise']) + '\n')
                    fout.write('\n')

                if status_memory['Deep Learning']:
                    dic = Counter(arrayTaskEmotionsDL)
                    dominant = max(dic, key=dic.get)
                    fout.write('Deep Learning sumary:\n')
                    fout.write('Dominant emotion in task: ' + dominant + '\n')
                    fout.write('Angry times: ' + str(dic['angry']) + '\n')
                    fout.write('Disgust times: ' + str(dic['disgust']) + '\n')
                    fout.write('Fear times: ' + str(dic['fear']) + '\n')
                    fout.write('Happy times: ' + str(dic['happy']) + '\n')
                    fout.write('Neutral times: ' + str(dic['neutral']) + '\n')
                    fout.write('Sad times: ' + str(dic['sad']) + '\n')
                    fout.write('Surprise times: ' + str(dic['surprise']) + '\n')
                    fout.write('\n')
                
                arrayTaskEmotionsDF = []
                arrayTaskEmotionsTL = []
                arrayTaskEmotionsDL = []
                lastLineTask = actualTask
                arrayTaskEmotionsDF.append(rowArray[3])
                arrayTaskEmotionsTL.append(rowArray[4])
                arrayTaskEmotionsDL.append(rowArray[5])



#Thread analysis
threadingActive = False

app = Flask(__name__)

@app.route('/result', methods= ['GET'])
def get_emotions():

    global txtEmotionDF
    global txtEmotionTL
    global txtEmotionDL

    global emotion_confidenceDF
    global emotion_confidenceTL
    global emotion_confidenceDL

    global status_memory

    FinalJSONTxt = "{"

    if not status_memory["Threshold mode"]:
        if status_memory['DeepFace']:
            neutralDF = emotion_confidenceDF[0]
            happyDF = emotion_confidenceDF[1]
            surpriseDF = emotion_confidenceDF[2]
            disgustDF = emotion_confidenceDF[3]
            fearDF = emotion_confidenceDF[4]
            sadDF = emotion_confidenceDF[5]
            angryDF = emotion_confidenceDF[6]

            DFJSON = {"DeepFace":{
                                "DominantEmotion":txtEmotionDF, 
                                "angry":str('%.4f'%(angryDF)).replace('.',','),
                                "disgust":str('%.4f'%(disgustDF)).replace('.',','),
                                "fear":str('%.4f'%(fearDF)).replace('.',','),
                                "happy":str('%.4f'%(happyDF)).replace('.',','),
                                "sad":str('%.4f'%(sadDF)).replace('.',','),
                                "surprise":str('%.4f'%(surpriseDF)).replace('.',','),
                                "neutral":str('%.4f'%(neutralDF)).replace('.',',')
                            }}
            
            DFJSONTxt = json.dumps(DFJSON)[1:-1]

            FinalJSONTxt = FinalJSONTxt + DFJSONTxt

            if status_memory['Transfer Learning'] or status_memory ['Deep Learning']:
                FinalJSONTxt = FinalJSONTxt + ',\n'

        if status_memory['Transfer Learning']:
            neutralTL = emotion_confidenceTL[0]
            happyTL = emotion_confidenceTL[1]
            surpriseTL = emotion_confidenceTL[2]
            disgustTL = emotion_confidenceTL[3]
            fearTL = emotion_confidenceTL[4]
            sadTL = emotion_confidenceTL[5]
            angryTL = emotion_confidenceTL[6]

            TLJSON = {"TransferLearning":{
                                "DominantEmotion":txtEmotionTL, 
                                "angry":str('%.4f'%(angryTL)).replace('.',','),
                                "disgust":str('%.4f'%(disgustTL)).replace('.',','),
                                "fear":str('%.4f'%(fearTL)).replace('.',','),
                                "happy":str('%.4f'%(happyTL)).replace('.',','),
                                "sad":str('%.4f'%(sadTL)).replace('.',','),
                                "surprise":str('%.4f'%(surpriseTL)).replace('.',','),
                                "neutral":str('%.4f'%(neutralTL)).replace('.',',')
                            }}
            

            TLJSONTxt = json.dumps(TLJSON)[1:-1]

            FinalJSONTxt = FinalJSONTxt + TLJSONTxt

            if status_memory ['Deep Learning']:
                FinalJSONTxt = FinalJSONTxt + ',\n'

        if status_memory['Deep Learning']:
            neutralDL = emotion_confidenceDL[0]
            happyDL = emotion_confidenceDL[1]
            surpriseDL = emotion_confidenceDL[2]
            disgustDL = emotion_confidenceDL[3]
            fearDL = emotion_confidenceDL[4]
            sadDL = emotion_confidenceDL[5]
            angryDL = emotion_confidenceDL[6]

            DLJSON = {"DeepLearning": {
                                "DominantEmotion":txtEmotionDL, 
                                "angry":str('%.4f'%(angryDL)).replace('.',','),
                                "disgust":str('%.4f'%(disgustDL)).replace('.',','),
                                "fear":str('%.4f'%(fearDL)).replace('.',','),
                                "happy":str('%.4f'%(happyDL)).replace('.',','),
                                "sad":str('%.4f'%(sadDL)).replace('.',','),
                                "surprise":str('%.4f'%(surpriseDL)).replace('.',','),
                                "neutral":str('%.4f'%(neutralDL)).replace('.',',')
                            }}
            
            DLJSONTxt = json.dumps(DLJSON)[1:-1]

            FinalJSONTxt = FinalJSONTxt + DLJSONTxt

        FinalJSONTxt = FinalJSONTxt + '}'
        print(FinalJSONTxt)
        FinalJSON = json.loads(FinalJSONTxt)
    
    else:
        time.sleep((status_memory["Threshold"]/2)/1000)
       
        actualDF = windowDF
        actualTL = windowTL
        actualDL = windowDL

    FinalJSON = {
        "DeepFace": averageEmotion(actualDF),
        "TransferLearning": averageEmotion(actualTL),
        "DeepLearning": averageEmotion(actualDL)
    }        

    if threadingActive:
        return FinalJSON
    else:
        return jsonify({
           "Analysing": "FALSE"
        })
        

def averageEmotion(list):

    names = ["neutral", "happy", "surprise", "disgust", "fear", "sad", "angry"]
    values = [0,0,0,0,0,0,0]

    for element in list:
        count = 0
        for value in element["analysis"]:
            values[count] += value
            count = count +1
    
    count = 0
    for element in values:
        values[count] = element/len(list)
        count = count +1
    
    dominant = names[np.argmax(values)]

    return {
                "DominantEmotion":dominant, 
                "angry":str('%.4f'%(values[6])).replace('.',','),
                "disgust":str('%.4f'%(values[3])).replace('.',','),
                "fear":str('%.4f'%(values[4])).replace('.',','),
                "happy":str('%.4f'%(values[1])).replace('.',','),
                "sad":str('%.4f'%(values[5])).replace('.',','),
                "surprise":str('%.4f'%(values[2])).replace('.',','),
                "neutral":str('%.4f'%(values[0])).replace('.',',')
            }



@app.route('/start', methods= ['GET'])
def startDF():
    global threadingActive
    if not status_memory['DeepFace'] and not status_memory['Transfer Learning'] and not status_memory['Deep Learning']:
        return jsonify({
            "Error": "No models active, analyse failed!"
        })
    threadingActive = True
    tStream = Thread(target=start_stream, args=(),daemon=True)
    tDF = Thread(target=start_analysis_deepFace, args=(), daemon=True)
    tTL = Thread(target=start_analysis_transfer, args=(), daemon=True)
    tDL = Thread(target=start_analysis_deepLearning, args=(), daemon=True)
    tlogs = Thread(target=start_log, args=(), daemon=True)
    tWebcamRecord = Thread(target=start_webcam_record, args=(), daemon=True)
    tScreenRecord = Thread(target=start_screen_record, args=(), daemon=True)
    tStream.start()
    if status_memory['DeepFace']:
        tDF.start()
    if status_memory['Transfer Learning']:    
        tTL.start()
    if status_memory['Deep Learning']:
        tDL.start()
    if status_memory['Logs']:
        tlogs.start()
    if status_memory['Screen']:
        tWebcamRecord.start()
    if status_memory['Video']:
        tScreenRecord.start()

    return jsonify({"Analysing": "TRUE"})

@app.route('/stop', methods= ['GET'])
def stopDF():
    global threadingActive
    global task
    global lasttask
    threadingActive = False
    task = 0
    lasttask = 0
    return jsonify({"Analysing": "FALSE"})

@app.route('/status', methods= ['GET'])
def sendStatus():
    return status_memory

@app.route('/models', methods=['POST'])
def updateModels():
    request_data = request.get_json()

    if 'DeepFace' not in request_data or 'Transfer' not in request_data or 'Deep' not in request_data:
        return jsonify({
            "Error":"Incorrect body data"
        })
    
    if not isinstance(status_memory['DeepFace'],bool) and not isinstance(status_memory['Transfer'],bool) and not isinstance(status_memory['Deep'],bool):
        return jsonify({
            "Error":"Incorrect value types"
        })

    deepFaceNewStatus = request_data['DeepFace']
    transferNewStatus = request_data['Transfer']
    deepLearningNewStatus = request_data['Deep']

    status_memory['DeepFace'] = deepFaceNewStatus
    status_memory['Transfer Learning'] = transferNewStatus
    status_memory['Deep Learning'] = deepLearningNewStatus

    f = open("Backend//Status//config.json", 'w+')
    f.write(json.dumps(status_memory))
    f.close()

    return status_memory

@app.route('/threshold', methods=['POST'])
def updateThreshold():
    request_data = request.get_json()

    newStatus = request_data['ThresholdMode']
    newTime = request_data['ThresholdTime']

    status_memory['Threshold mode'] = newStatus
    status_memory['Threshold'] = newTime

    f = open("Backend//Status//config.json", 'w+')
    f.write(json.dumps(status_memory))
    f.close()

    return status_memory

@app.route('/video', methods=['POST'])
def updateVideoSettings():
    request_data = request.get_json()

    newFolder = request_data['VideoFolder']
    newStatus = request_data['Video']

    if newFolder != '':
        status_memory['Video Folder'] = newFolder

    status_memory['Video'] = newStatus

    f = open("Backend//Status//config.json", 'w+')
    f.write(json.dumps(status_memory))
    f.close()

    return status_memory

@app.route('/screen', methods=['POST'])
def updateScreenSettings():
    request_data = request.get_json()

    newFolder = request_data['ScreenFolder']
    newStatus = request_data['Screen']

    if newFolder != '':
        status_memory['Screen Folder'] = newFolder

    status_memory['Screen'] = newStatus

    f = open("Backend//Status//config.json", 'w+')
    f.write(json.dumps(status_memory))
    f.close()

    return status_memory

@app.route('/log', methods=['POST'])
def updateLogSettings():

    request_data = request.get_json()

    newFolder = request_data['LogsFolder']
    newStatus = request_data['Logs']

    if newFolder != '':
        status_memory['Logs Folder'] = newFolder

    status_memory['Logs'] = newStatus

    f = open("Backend//Status//config.json", 'w+')
    f.write(json.dumps(status_memory))
    f.close()

    return status_memory

@app.route('/webcam', methods=['POST'])
def updateWebcam():
    global stream

    stream.release()

    request_data = request.get_json()

    newWebcam = request_data['Webcam']

    if type(newWebcam) == str:
        newWebcam = int(newWebcam)

    status_memory['Webcam'] = newWebcam

    f = open("Backend//Status//config.json", 'w+')
    f.write(json.dumps(status_memory))
    f.close()

    stream = cv2.VideoCapture(status_memory['Webcam'])

    return status_memory

@app.route('/screencam', methods=['POST'])
def updateScreencam():

    request_data = request.get_json()

    newScreencam = request_data['Screen']

    if type(newScreencam) == str:
        newScreencam = int(newScreencam)

    status_memory['Screencam'] = newScreencam

    f = open("Backend//Status//config.json", 'w+')
    f.write(json.dumps(status_memory))
    f.close()

    return status_memory

@app.route('/task', methods=['GET'])
def startTask():
    
    global task
    global lasttask

    if task != 0:
        return jsonify({"Task":task})
    else:
        task = lasttask + 1
        return jsonify({"Task": task})
    
@app.route('/notask', methods=['GET'])
def endTask():
    global task
    global lasttask

    if task == 0:
        return jsonify({"Task":0})
    else:
        lasttask = task
        task = 0
        return jsonify({"Task": 0})



if __name__ == "__main__":
    app.run(debug=True)