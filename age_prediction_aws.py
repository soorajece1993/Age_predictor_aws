import numpy as np
import cv2
import boto3
import json
import csv
import time
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (255, 120, 0)
fontSize = 0.8
with open("new_user_credentials.csv",'r') as input:
    next(input)
    reader=csv.reader(input)
    print(reader)

    for line in reader:
        access_key_id=line[2]
        secret_key_id=line[3]

# Rekognition Detect faces
def detect_faces(photo):

    # client=boto3.client('rekognition')
    region = "eu-west-1"
    client = boto3.client("rekognition", aws_access_key_id=access_key_id,
                          aws_secret_access_key=secret_key_id, region_name=region)

    response = client.detect_faces(
        Image={
            'Bytes': photo
        },
        Attributes=[
            'ALL'
        ]
    )
    return response

cam = cv2.VideoCapture(0)
cv2.namedWindow("AGE Predictor")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cam.read()
    cv2.imshow("AGE Predictor", frame)


    print(frame.shape)
    out.write(frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        resp=(detect_faces(cv2.imencode('.jpg', frame)[1].tostring()))
        # print(resp)
        Age = resp['FaceDetails'][0]['AgeRange']
        print(str(Age))
        cv2.putText(frame, 'AGE  :  ' + str(Age), (10, 90), font, fontSize, fontColor, 2)
        cv2.imshow("AGE Predictor", frame)
        for i in range(0, 40):
            out.write(frame)
        cv2.waitKey(2000)
        # time.sleep(2)





out.release()

# Release control of the webcam and close window
cam.release()
cv2.destroyAllWindows()