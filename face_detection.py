import cv2 as cv
import argparse


parser = argparse.ArgumentParser(description='Script for face detection with webcam.')
parser.add_argument('--cascade', help='Path to input haar_cascade file.')
args = parser.parse_args()
# Initializing the VideoCapture Variable
cap =cv.VideoCapture(0)

# Reading in the Haar Cascade file
haar_cascade = cv.CascadeClassifier(cv.samples.findFile(args.cascade))

if haar_cascade is None:
    print('Could not open or find the haar_cascade file, Check your path and try again: ', args.cascade)
    exit(0)

# Getting video frame by frame
while True:
    isTrue, frame = cap.read()

    # Converting each frame from Colored(BGR) to GrayScale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    

    
    # Detect face(s) in GrayScale frame
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # print(f'Number of faces found = {len(faces_rect)}')

    # Draw rectangle over detected faces
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv.imshow('Detected Faces', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break


cap.release()
cv.destroyAllWindows()