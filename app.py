
# import neccessary libraries
import cv2
import sys
import time
import math
import numpy as np
import random as rng
from scipy import stats
import kociemba
from datetime import datetime

# import functions that are required
from rotate import *


# method to concate all the faces in a way so that it can be given to kociemba module
def concat(up_face,right_face,front_face,down_face,left_face,back_face):
    # solution = [up_face,right_face,front_face,down_face,left_face,back_face]
    solution = np.concatenate((up_face, right_face), axis=None)
    solution = np.concatenate((solution, front_face), axis=None)
    solution = np.concatenate((solution, down_face), axis=None)
    solution = np.concatenate((solution, left_face), axis=None)
    solution = np.concatenate((solution, back_face), axis=None)
    # print(solution)
    return solution

# method to detect faces from the cube
def detect_face(bgr_image_input):

    # convert  image to gray
    gray = cv2.cvtColor(bgr_image_input,cv2.COLOR_BGR2GRAY)

    # defining kernel for morphological operations using ellipse structure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    #cv2.imshow('gray',gray)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # gray = cv2.Canny(bgr_image_input,50,100)
    #cv2.imshow('gray',gray)
    # adjusting threshold to get countours easily
    # these needs to be changed based on the lighting condition you have and the environment you are using
    gray = cv2.adaptiveThreshold(gray,37,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,19,0)
    #cv2.imshow('gray',gray)
    #cv2.imwrite()

    # for finding contours you can also use canny functions that is available in cv2 but for my environment
    # I found gray was working better
    try:
        # get contours from the image after applying morphological operations
         _, contours, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    except:
         contours, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)


    i = 0
    contour_id = 0
    #print(len(contours))
    count = 0
    blob_colors = []
    for contour in contours:
        # get area of contours , obviously we don't want every contour in our image
        A1 = cv2.contourArea(contour)
        contour_id = contour_id + 1

        if A1 < 3000 and A1 > 1000:
            perimeter = cv2.arcLength(contour, True)

            # after checking the area we will estimate the epsilon structure
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # this is a just in case scenario
            hull = cv2.convexHull(contour)
            if cv2.norm(((perimeter / 4) * (perimeter / 4)) - A1) < 150:
                #if cv2.ma
                count = count + 1

                # get co ordinates of the contours in the cube
                x, y, w, h = cv2.boundingRect(contour)
                #cv2.rectangle(bgr_image_input, (x, y), (x + w, y + h), (0, 255, 255), 2)
                #cv2.imshow('cutted contour', bgr_image_input[y:y + h, x:x + w])
                val = (50*y) + (10*x)

                # get mean color of the contour
                blob_color = np.array(cv2.mean(bgr_image_input[y:y+h,x:x+w])).astype(int)

                # below code is to convert bgr color to hsv values so that i can use it in the if conditions
                # even bgr can be used but in my case hsv was giving better results
                blue = blob_color[0]/255
                green = blob_color[1]/255
                red = blob_color[2]/255
                
                cmax = max(red,blue,green)
                cmin = min(red,blue,green)
                diff = cmax - cmin
                hue = -1
                saturation = -1

                if (cmax == cmin):
                        hue = 0 


                elif (cmax == red):
                        hue = (60 *((green - blue)/ diff) + 360) % 360; 


                elif (cmax == green):
                        hue = (60 * ((blue - red) / diff) + 120) % 360; 


                elif (cmax == blue): 
                        hue = (60 * ((red - green) / diff) + 240) % 360; 

                if (cmax == 0):
                        saturation = 0; 
                else:
                        saturation = (diff / cmax) * 100; 

                value = cmax * 100;

                #print(hue,saturation,value)
                #exit()
                
                blob_color[0] , blob_color[1], blob_color[2] = hue,saturation,value	
           
                #print(blob_color)
                cv2.drawContours(bgr_image_input,[contour],0,(255, 255, 0),2)
                cv2.drawContours(bgr_image_input, [approx], 0, (255, 255, 0), 2)
                blob_color = np.append(blob_color, val)
                blob_color = np.append(blob_color, x)
                blob_color = np.append(blob_color, y)
                blob_color = np.append(blob_color, w)
                blob_color = np.append(blob_color, h)
                blob_colors.append(blob_color)
    if len(blob_colors) > 0:
        blob_colors = np.asarray(blob_colors)
        blob_colors = blob_colors[blob_colors[:, 4].argsort()]
    face = np.array([0,0,0,0,0,0,0,0,0])
    if len(blob_colors) == 9:
        #print(blob_colors)
        for i in range(9):
            #print(blob_colors[i])
            # assign values to blob_color and faces based on the hsv values
            if 230<=blob_colors[i][0]  and blob_colors[i][1] <=20 and 20 <= blob_colors[i][2] <= 60:
                blob_colors[i][3] = 1
                face[i] = 1
                #print('black detected')
            elif 40<= blob_colors[i][0] <= 80 and 60 <= blob_colors[i][1] <= 90 and 70<= blob_colors[i][2] <= 110:
                blob_colors[i][3] = 2
                face[i] = 2
                #print('yellow detected')
            elif 190 <= blob_colors[i][0] <= 225 and 55 <= blob_colors[i][1] <= 95 and 35<= blob_colors[i][2]<=75:
                blob_colors[i][3] = 3
                face[i] = 3
                #print('blue detected')
            elif 100 <= blob_colors[i][0] <= 150 and 25 <= blob_colors[i][1] <= 50 and 40<= blob_colors[i][2]<=80:
                blob_colors[i][3] = 4
                face[i] = 4
                #print('green detected')
            elif 325 <= blob_colors[i][0] <= 365 and 50 <= blob_colors[i][1] <= 80 and 45<= blob_colors[i][2]<=75:
                blob_colors[i][3] = 5
                face[i] = 5
                #print('red detected')
            elif blob_colors[i][0] <= 30 and 65<= blob_colors[i][1] <= 90 and 60<= blob_colors[i][2]<=90:
                blob_colors[i][3] = 6
                face[i] = 6
                #print('orange detected')
        #print(face)
        if np.count_nonzero(face) == 9:
            #print(face)
            #print (blob_colors)
            return face, blob_colors
        else:
            return [0,0], blob_colors
    else:
        return [0,0,0], blob_colors
        #break

def find_face(video,videoWriter,uf,rf,ff,df,lf,bf,text = ""):
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()


        # assinging values to face and blob colors based on the detect_face method
        face, blob_colors = detect_face(bgr_image_input)
        bgr_image_input = cv2.putText(bgr_image_input, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 5:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                # print(final_face)
                uf = np.asarray(uf)
                ff = np.asarray(ff)
                detected_face = np.asarray(detected_face)
                #print(np.array_equal(detected_face, tf))
                #print(np.array_equal(detected_face, ff))
                faces = []
                if np.array_equal(detected_face, uf) == False and np.array_equal(detected_face, ff) == False and np.array_equal(detected_face, bf) == False and np.array_equal(detected_face, df) == False and np.array_equal(detected_face, lf) == False and np.array_equal(detected_face, rf) == False:
                    return detected_face
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


# main method
def main():
    up_face = [0, 0]
    front_face = [0, 0]
    left_face = [0, 0]
    right_face = [0, 0]
    down_face = [0, 0]
    back_face = [0, 0]

    # initialising web cam for recording
    video = cv2.VideoCapture(0)
    
    #video = cv2.VideoCapture('http://192.168.43.1:8080/video')
    is_ok, bgr_image_input = video.read()
    broke = 0
    

    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    h1 = bgr_image_input.shape[0]
    w1 = bgr_image_input.shape[1]
    faces = []
    
    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fname = "OUTPUT5.avi"
        fps = 24.0
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, (w1, h1))
    except:
        print("Error: can't create output video: %s" % fname)
        sys.exit()
    
    while True:
        is_ok, bgr_image_input = video.read()
        if not is_ok:
            break

        # below get all faces using above to functions
        while True:
            #print("Show Front Face")
            front_face = find_face(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face, text="Show Front Face")
            mf = front_face[0,4]
            print(front_face)
            print(type(front_face))
            print(mf)
            #print("Show Up Face")
            #time.sleep(2)
            up_face = find_face(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face, text="Show Top Face")
            start_time = datetime.now()
            # this loop is to stay same side for 3 seconds so that by mistake wrong color will not be detected
            # due to lighting factor
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Down Face", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    videoWriter.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            mu = up_face[0, 4]
            print(up_face)
            print(mu)
            #print("Show Down Face")
            #time.sleep(2)
            down_face = find_face(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face, text="Show Down Face")
            start_time = datetime.now()
            while True:

                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Right Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    videoWriter.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            md = down_face[0, 4]
            print(down_face)
            print(md)
            #print("Show Right Face")
            #time.sleep(2)
            right_face = find_face(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face, text="Show Right Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Left Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    videoWriter.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            mr = right_face[0, 4]
            print(right_face)
            print(mr)
            #print("Show Left Face")
            #time.sleep(2)
            left_face = find_face(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face, text="Show Left Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Back Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    videoWriter.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            ml = left_face[0, 4]
            print(left_face)
            print(ml)
            #print("Show Back Face")
            #time.sleep(2)
            back_face = find_face(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face, text="Show Back Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Front Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    videoWriter.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            mb = back_face[0, 4]
            print(back_face)
            #time.sleep(2)
            print(mb)

            # append all the faces in the order so that it can be given to kociemba module
            solution = concat(up_face,right_face,front_face,down_face,left_face,back_face)
            #print(solution)
            cube_solved = [mu, mu, mu, mu, mu, mu, mu, mu, mu, mr, mr, mr, mr, mr, mr, mr, mr, mr, mf, mf, mf, mf, mf,
                           mf, mf, mf, mf, md, md, md, md, md, md, md, md, md, ml, ml, ml, ml, ml, ml, ml, ml, ml, mb,
                           mb, mb, mb, mb, mb, mb, mb, mb]
            if (concat(up_face, right_face, front_face, down_face, left_face, back_face) == cube_solved).all():
                # print("CUBE IS SOLVED")
                is_ok, bgr_image_input = video.read()
                bgr_image_input = cv2.putText(bgr_image_input, "CUBE ALREADY SOLVED", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                videoWriter.write(bgr_image_input)
                cv2.imshow("Output Image", bgr_image_input)
                key_pressed = cv2.waitKey(1) & 0xFF
                if key_pressed == 27 or key_pressed == ord('q'):
                    break
                time.sleep(5)
                break


            # assigning respective values to the faces
            ''' F -------> Front face
                R -------> Right face
                B -------> Back face
                L -------> Left face
                U -------> Up face
                D -------> Down face'''
            final_str = ''
            for val in range(len(solution)):
                if solution[val] == mf:
                    final_str = final_str + 'F'
                elif solution[val] == mr:
                    final_str = final_str + 'R'
                elif solution[val] == mb:
                    final_str = final_str + 'B'
                elif solution[val] == ml:
                    final_str = final_str + 'L'
                elif solution[val] == mu:
                    final_str = final_str + 'U'
                elif solution[val] == md:
                    final_str = final_str + 'D'

            print(final_str)
            try:
                solved = kociemba.solve(final_str)
                print(solved)
                break
            except:
                up_face = [0, 0]
                front_face = [0, 0]
                left_face = [0, 0]
                right_face = [0, 0]
                down_face = [0, 0]
                back_face = [0, 0]

        if broke == 1:
            break
        # spliting the steps and calling respective functions so that arrows can be displayed accordingly
        # below methods are available in the rotate.py file
        # in steps letter like R F B D L U indicate right, front, back , down faces to be rotated clockwise
        # if there is 2 after the letter they should be rotate twice and if there is " ' " like R' then the respective
        # face should be rotated anti clock wise
        steps = solved.split()
        for step in steps:
            if step == "R":
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "R'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_ccw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "R2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "L":
                [up_face, right_face, front_face, down_face, left_face, back_face] = left_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "L'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = left_ccw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "L2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = left_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = left_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "F":
                [up_face, right_face, front_face, down_face, left_face, back_face] = front_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "F'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = front_ccw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "F2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = front_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = front_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "B":
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "B'":
                #print(up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_ccw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
            elif step == "B2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "U":
                [up_face, right_face, front_face, down_face, left_face, back_face] = up_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "U'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = up_ccw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "U2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = up_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = up_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "D":
                [up_face, right_face, front_face, down_face, left_face, back_face] = down_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "D'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = down_ccw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "D2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = down_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = down_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
                #print(concat(up_face, right_face, front_face, down_face, left_face, back_face))

        #after solving all we just need to check the strings and display message accordingly
        cube_solved = [mu, mu, mu, mu, mu, mu, mu, mu, mu, mr, mr, mr, mr, mr, mr, mr, mr, mr, mf, mf, mf, mf, mf, mf, mf, mf, mf, md, md, md, md, md, md, md, md, md, ml, ml, ml, ml, ml, ml, ml, ml, ml, mb, mb, mb, mb, mb, mb, mb, mb, mb]
        if (concat(up_face, right_face, front_face, down_face, left_face, back_face) == cube_solved).all():
            #print("CUBE IS SOLVED")
            is_ok, bgr_image_input = video.read()
            bgr_image_input = cv2.putText(bgr_image_input, "CUBE SOLVED", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            videoWriter.write(bgr_image_input)
            cv2.imshow("Output Image", bgr_image_input)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == 27 or key_pressed == ord('q'):
                break
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 5:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "CUBE SOLVED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    videoWriter.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            break
        #print(front_face)
        #print(up_face)

        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        # print(count)
        # print(blob_color)
        # print(face)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


if __name__ == "__main__":
    main()
