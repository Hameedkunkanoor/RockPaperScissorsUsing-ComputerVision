import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
RCP_dict = {0: "Rock", 1: "Paper", 2: "Scissors"}

    
cap = cv2.VideoCapture(0)


model=load_model('RockPaperScissors.h5')
while True:
        
        keras.clear_session()
        ret, frame = cap.read()
        if not ret:
            break
        #frame=frame.resize(1,300,200,3)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped_img = np.expand_dims(cv2.resize(gray, (200, 300)), -1)

        rr=gray.resize(300,200,3)
        prediction = model.predict(rr,steps=1)
        #maxindex = np.argmax(prediction)
        print(prediction)
        #cv2.putText(frame, prediction, (40+20, 120-60), 1, (255, 255, 255), 2, cv2.LINE_AA)
        objects = (['Rock', 'Paper', 'Scissors'])
        index = np.arange(len(objects))
        MakingPositive=lambda a: (abs(a)+a)/2
            
        cv2.imshow('ROCKPAPERSCISSORS', cv2.resize(frame,(600,500),interpolation = cv2.INTER_CUBIC))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break

cap.release()
cv2.destroyAllWindows()