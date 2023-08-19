import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

camw = 640
camh = 480

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) 
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    silhouette = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]
    newimage = image
    face = image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:        
            xcoords = [handmark.x for handmark in face_landmarks.landmark]
            ycoords = [handmark.y for handmark in face_landmarks.landmark]
            sxcoords = [int(xcoords[i] * camw) for i in silhouette]
            sycoords = [int(ycoords[i] * camh) for i in silhouette]
            rex = int(xcoords[33] * camw)
            rey = int(ycoords[33] * camh)           
            lex = int(xcoords[263] * camw)
            ley = int(ycoords[263] * camh)
            midx = int((rex + lex)/2)
            midy = int((rey + ley)/2)
            angle = math.atan((rey-ley)/(rex-lex))
            deg = int(360 + (angle * (180/3.14159))) % 360
            print(deg)
            M = cv2.getRotationMatrix2D((midx, midy), deg, 1.0)
            print(M)
            sxx = [int(M[0][0] * sxcoords[i] + M[0][1] * sycoords[i] + M[0][2]) for i in range(len(sycoords))]
            sxy = [int(M[1][0] * sxcoords[i] + M[1][1] * sycoords[i] + M[1][2]) for i in range(len(sycoords))]
            newimage = cv2.warpAffine(image, M, (640, 480))            
            minx = min(sxx)
            maxx = max(sxx)
            miny = min(sxy)
            maxy = max(sxy)
            face = newimage[miny:maxy, minx:maxx]
            cv2.rectangle(newimage, (minx, miny), (maxx, maxy), (0,0,255), 3)

    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Face Mesh1', cv2.flip(newimage, 1))    
    cv2.imshow('MediaPipe Face Mesh2', cv2.flip(face, 1))    

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
