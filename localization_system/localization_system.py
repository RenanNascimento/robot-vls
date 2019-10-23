import os
import tensorflow as tf
import numpy as np
import math
import cv2
import ast
import socket
import urllib.request

from tensorflow import keras

def createSocketCon():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('192.168.0.101', 8080))
    return client

def createModel():
    # load json and create model
    json_file = open('mlp_model/mlp_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json, custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform})
    # load weights into new model
    loaded_model.load_weights("mlp_model/mlp_model.h5")
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    loaded_model.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    return loaded_model

#calculate angle
def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

def serving_input_fn():
    x = tf.placeholder(dtype=tf.float32, shape=[1, 28, 28], name='x')
    inputs = {'x': x }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def main():

    url = 'http://192.168.0.100:8080/shot.jpg'

    # Create the Estimator
    mlp_model = createModel()

    # Create socket connection
    client = createSocketCon()

    #dictionary of all contours
    contours = {}
    #array of edges of polygon
    approx = []
    #scale of the text
    scale = 2
    #camera
    #cap = cv2.VideoCapture(0)
    #print("press q to exit")

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while True:
        tri_detected = False
        rect_detected = False
        
        #Capture frame-by-frame
        with urllib.request.urlopen(url) as url_frame:
            frame = np.array(bytearray(url_frame.read()), dtype=np.uint8)
        frame = cv2.imdecode(frame, -1)
        frame_height, frame_width, frame_channels = frame.shape
        cv2.circle(frame, (int(frame_width/2), int(frame_height/2)), 5, (0, 255, 255), -1)
        
        #ret, frame = cap.read()
        if True:
            #grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Canny
            canny = cv2.Canny(gray,80,240,3)

            #contours
            canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0,len(contours)):
                #approximate the contour with accuracy proportional to
                #the contour perimeter
                approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

                #Skip small or non-convex objects
                if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
                    continue

                #triangle
                if(len(approx)==3):
                    tri_detected = True
                    (x_tri, y_tri), radius_tri = cv2.minEnclosingCircle(contours[i])
                    center_tri = (int(x_tri), int(y_tri))
                    cv2.circle(frame, center_tri, int(radius_tri), (0,0,255), 2)
                    #cv2.circle(frame, center_tri, 5, (0, 255, 0), -1)

                #rect
                if(len(approx)>=4 and len(approx)<=6):
                    rect_detected = True
                    #nb vertices of a polygonal curve
                    vtc = len(approx)
                    #get cos of all corners
                    cos = []
                    for j in range(2,vtc+1):
                        cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
                    #sort ascending cos
                    cos.sort()
                    #get lowest and highest
                    mincos = cos[0]
                    maxcos = cos[-1]

                    #Use the degrees obtained above and the number of vertices
                    #to determine the shape of the contour
                    x,y,w,h = cv2.boundingRect(contours[i])
                    # Rotated Rectangle
                    rect = cv2.minAreaRect(contours[i])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    square_like = False
                    if(w > h):
                        ratio = w/float(h)
                    else:
                        ratio = h/float(w)
                    if(vtc==4 and ratio <= 1.2):
                        cv2.drawContours(frame,[box],0,(0,255,0),2)
                        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                        # get roi
                        roi = frame[y:y+h, x:x+w]

                        # rotate roi
                        rows, cols, n_channels = roi.shape
                        M = cv2.getRotationMatrix2D((cols/2,rows/2),rect[2],1)
                        dst = cv2.warpAffine(roi,M,(cols,rows))

                        center = {'x': rect[0][0], 'y': rect[0][1]}
                        #cv2.circle(frame, (int(center['x']), int(center['y'])), 5, (0, 0, 255), -1)
                        dim = {'width': int(rect[1][0]), 'height': int(rect[1][1])}
                        point = {'x': int(center['x'] - dim['width']/2), 'y': int(center['y'] - dim['height']/2)}
                        point_roi = {'x': point['x']-x, 'y': point['y']-y}
                        roi_rotated = dst[point_roi['y']:point_roi['y']+dim['height'], point_roi['x']:point_roi['x']+dim['width']]
                        
                        if roi_rotated.shape[0] > 0 and roi_rotated.shape[1] > 0:
                            roi_rotated_risized = cv2.resize(roi_rotated, (28, 28))
                            roi_rotated_gray = cv2.cvtColor(roi_rotated_risized, cv2.COLOR_BGR2GRAY)
                            # binary
                            roi_rotated_gray = cv2.adaptiveThreshold(roi_rotated_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
                            #cv2.imshow('roi_rotated', roi_rotated_gray)
                            ret1,thresh1 = cv2.threshold(roi_rotated_gray,127,255,cv2.THRESH_BINARY)

                            prediction = mlp_model.predict(thresh1.reshape(1, 28, 28))
                            prop = list(prediction[0])
                            
                            if not tri_detected:
                                num_predicted = str('{ "marker": ' + str(prop.index(max(prop))) +' }').encode()
                                client.send(num_predicted)

            if tri_detected and rect_detected:
                center_rect = (int(center['x']), int(center['y']))
                cv2.line(frame, center_tri, center_rect, (0, 255, 0), 3)
                x_axis = - (center_tri[1] - center_rect[1]) + center_rect[0]
                y_axis = (center_tri[0] - center_rect[0]) + center_rect[1]
                point_rot = (int(x_axis), int(y_axis))
                #cv2.circle(frame, point_rot, 5, (0, 255, 0), -1)
                cv2.line(frame, point_rot, center_rect, (0, 0, 255), 3)
                
                robot_pos = (int(frame_width/2), int(frame_height/2))
                
                p1 = np.asarray(center_rect)
                p2 = np.asarray(center_tri)
                p3 = np.asarray(point_rot)
                p4 = np.asarray(robot_pos)
                dist_x = np.linalg.norm(np.cross(p2-p1, p1-p4))/np.linalg.norm(p2-p1)
                dist_y = np.linalg.norm(np.cross(p3-p1, p1-p4))/np.linalg.norm(p3-p1)
                
                dist_x = round(dist_x, 2)
                dist_y = round(dist_y, 2)
                
                if p1[0] - p4[0] < 0:
                    dist_x = dist_x * -1
                if p4[1] - p1[1] < 0:
                    dist_y = dist_y * -1
            
                pose_estimated = str('{ "pose": { "marker": '+ str(prop.index(max(prop))) +', "dist_x_pixels": '+ str(dist_x) +', "dist_y_pixels": '+ str(dist_y) +' } }').encode()
                client.send(pose_estimated)
            
            #Display the resulting frame
            #out.write(frame)
            frame_height, frame_width, frame_channels = frame.shape
            cv2.imshow('frame', cv2.resize(frame, (int(frame_width/2), int(frame_height/2))))
            #cv2.imshow('roi', cv2.resize(roi, (int(frame_width/2), int(frame_height/2))))
            if cv2.waitKey(33) == ord('a'):
                break

    #When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    client.close()

main()