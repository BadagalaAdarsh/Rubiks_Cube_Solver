# import neccessary libraries
import sys
import time
from datetime import datetime
import cv2
import kociemba
import numpy as np
from scipy import stats


# method to concate all the faces in a way so that it can be given to kociemba module
def face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face):
    # solution = [up_face,right_face,front_face,down_face,left_face,back_face]
    solution = np.concatenate((up_face, right_face, front_face, down_face, left_face, back_face))
    # print(solution)
    return solution


# method to detect faces from the cube
def face_detection_in_cube(bgr_image_input):
    # convert  image to gray
    gray = cv2.cvtColor(bgr_image_input, cv2.COLOR_BGR2GRAY)

    # defining kernel for morphological operations using ellipse structure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('gray',gray)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # gray = cv2.Canny(bgr_image_input,50,100)
    # cv2.imshow('gray',gray)
    # adjusting threshold to get countours easily
    # these needs to be changed based on the lighting condition you have and the environment you are using
    gray = cv2.adaptiveThreshold(gray, 37, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 0)
    # cv2.imshow('gray',gray)
    # cv2.imwrite()

    # for finding contours you can also use canny functions that is available in cv2 but for my environment
    # I found gray was working better
    try:
        # get contours from the image after applying morphological operations
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    except:
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    i = 0
    contour_id = 0
    # print(len(contours))
    count = 0
    colors_array = []
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
                # if cv2.ma
                count = count + 1

                # get co ordinates of the contours in the cube
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(bgr_image_input, (x, y), (x + w, y + h), (0, 255, 255), 2)
                # cv2.imshow('cutted contour', bgr_image_input[y:y + h, x:x + w])
                val = (50 * y) + (10 * x)

                # get mean color of the contour
                color_array = np.array(cv2.mean(bgr_image_input[y:y + h, x:x + w])).astype(int)

                # below code is to convert bgr color to hsv values so that i can use it in the if conditions
                # even bgr can be used but in my case hsv was giving better results
                blue = color_array[0] / 255
                green = color_array[1] / 255
                red = color_array[2] / 255

                cmax = max(red, blue, green)
                cmin = min(red, blue, green)
                diff = cmax - cmin
                hue = -1
                saturation = -1

                if (cmax == cmin):
                    hue = 0


                elif (cmax == red):
                    hue = (60 * ((green - blue) / diff) + 360) % 360;


                elif (cmax == green):
                    hue = (60 * ((blue - red) / diff) + 120) % 360;


                elif (cmax == blue):
                    hue = (60 * ((red - green) / diff) + 240) % 360;

                if (cmax == 0):
                    saturation = 0;
                else:
                    saturation = (diff / cmax) * 100;

                value = cmax * 100;

                # print(hue,saturation,value)
                # exit()

                color_array[0], color_array[1], color_array[2] = hue, saturation, value

                # print(color_array)
                cv2.drawContours(bgr_image_input, [contour], 0, (255, 255, 0), 2)
                cv2.drawContours(bgr_image_input, [approx], 0, (255, 255, 0), 2)
                color_array = np.append(color_array, val)
                color_array = np.append(color_array, x)
                color_array = np.append(color_array, y)
                color_array = np.append(color_array, w)
                color_array = np.append(color_array, h)
                colors_array.append(color_array)
    if len(colors_array) > 0:
        colors_array = np.asarray(colors_array)
        colors_array = colors_array[colors_array[:, 4].argsort()]
    face = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    if len(colors_array) == 9:
        # print(colors_array)
        for i in range(9):
            # print(colors_array[i])
            # assign values to color_array and faces based on the hsv values
            if 230 <= colors_array[i][0] and colors_array[i][1] <= 20 and 20 <= colors_array[i][2] <= 60:
                colors_array[i][3] = 1
                face[i] = 1
                # print('black detected')
            elif 40 <= colors_array[i][0] <= 80 and 60 <= colors_array[i][1] <= 90 and 70 <= colors_array[i][2] <= 110:
                colors_array[i][3] = 2
                face[i] = 2
                # print('yellow detected')
            elif 190 <= colors_array[i][0] <= 225 and 55 <= colors_array[i][1] <= 95 and 35 <= colors_array[i][2] <= 75:
                colors_array[i][3] = 3
                face[i] = 3
                # print('blue detected')
            elif 100 <= colors_array[i][0] <= 150 and 25 <= colors_array[i][1] <= 50 and 40 <= colors_array[i][2] <= 80:
                colors_array[i][3] = 4
                face[i] = 4
                # print('green detected')
            elif 325 <= colors_array[i][0] <= 365 and 50 <= colors_array[i][1] <= 80 and 45 <= colors_array[i][2] <= 75:
                colors_array[i][3] = 5
                face[i] = 5
                # print('red detected')
            elif colors_array[i][0] <= 30 and 65 <= colors_array[i][1] <= 90 and 60 <= colors_array[i][2] <= 90:
                colors_array[i][3] = 6
                face[i] = 6
                # print('orange detected')
        # print(face)
        if np.count_nonzero(face) == 9:
            # print(face)
            # print (colors_array)
            return face, colors_array
        else:
            return [0, 0], colors_array
    else:
        return [0, 0, 0], colors_array
        # break




# method to rotate particular face clock wise
# it will the return the values of the face after the rotation is completed
def rotate_clock_wise(face):
    temp = np.copy(face)
    temp[0, 0], temp[0,1], temp[0,2], temp[0,3], temp[0,4], temp[0,5],temp[0,6],temp[0,7], temp[0,8] = face[0, 6],face[0, 3],face[0, 0],face[0, 7],face[0, 4],face[0, 1],face[0, 8],face[0, 5],face[0, 2]
    return temp

# method to rotate particular face counter clock wise or anti clock wise
# it will the return the values of the face after the rotation is completed
def rotate_counter_clock_wise(face):
    temp = np.copy(face)
    temp[0, 8],temp[0, 7],temp[0, 6],temp[0, 5],temp[0, 4],temp[0, 3],temp[0, 2],temp[0, 1], temp[0, 0] = face[0, 6],face[0, 3],face[0, 0],face[0, 7],face[0, 4],face[0, 1],face[0, 8],face[0, 5],face[0, 2]
    return temp

# method to rotate right face clock wise
def right_face_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: R Clockwise")
    temp = np.copy(front_face)
    front_face[0, 2] = down_face[0, 2]
    front_face[0, 5] = down_face[0, 5]
    front_face[0, 8] = down_face[0, 8]
    down_face[0, 2] = back_face[0, 6]
    down_face[0, 5] = back_face[0, 3]
    down_face[0, 8] = back_face[0, 0]
    back_face[0, 0] = up_face[0, 8]
    back_face[0, 3] = up_face[0, 5]
    back_face[0, 6] = up_face[0, 2]
    up_face[0, 2] = temp[0, 2]
    up_face[0, 5] = temp[0, 5]
    up_face[0, 8] = temp[0, 8]
    right_face = rotate_clock_wise(right_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        # get current face date using detect face method
        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # if detected face and the actual face after rotaion is same return the updated faces
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face

                # else display the arrow by calculation the centroid using the coordinates
                # that are available in color_array
                # this same logic applies to all the methods below
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[8]
                    centroid2 = colors_array[2]
                    point1 = (centroid1[5]+(centroid1[7]//2), centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2), centroid2[6]+(centroid2[8]//2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

# method to rotate right face in counter clock wise direction
def right_counter_clock_wise(video, videoWriter, up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: R CounterClockwise")
    temp = np.copy(front_face)
    front_face[0, 2] = up_face[0, 2]
    front_face[0, 5] = up_face[0, 5]
    front_face[0, 8] = up_face[0, 8]
    up_face[0, 2] = back_face[0, 6]
    up_face[0, 5] = back_face[0, 3]
    up_face[0, 8] = back_face[0, 0]
    back_face[0, 0] = down_face[0, 8]
    back_face[0, 3] = down_face[0, 5]
    back_face[0, 6] = down_face[0, 2]
    down_face[0, 2] = temp[0, 2]
    down_face[0, 5] = temp[0, 5]
    down_face[0, 8] = temp[0, 8]
    right_face = rotate_counter_clock_wise(right_face)
    # front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # if detected face and actual face after rotation is same return the update faces
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face

                # else display the arrow by calculation the centroid using the coordinates
                # that are available in color_array
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[2]
                    centroid2 = colors_array[8]
                    point1 = (centroid1[5]+(centroid1[7]//2), centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2), centroid2[6]+(centroid2[8]//2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

# method to rotate left face clock wise direction
def left_face_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: L Clockwise")
    temp = np.copy(front_face)
    front_face[0, 0] = up_face[0, 0]
    front_face[0, 3] = up_face[0, 3]
    front_face[0, 6] = up_face[0, 6]
    up_face[0, 0] = back_face[0, 8]
    up_face[0, 3] = back_face[0, 5]
    up_face[0, 6] = back_face[0, 2]
    back_face[0, 2] = down_face[0, 6]
    back_face[0, 5] = down_face[0, 3]
    back_face[0, 8] = down_face[0, 0]
    down_face[0, 0] = temp[0, 0]
    down_face[0, 3] = temp[0, 3]
    down_face[0, 6] = temp[0, 6]
    left_face = rotate_clock_wise(left_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # if detected face and actual face after rotation is same then return updated faces
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face

                # else display the arrow by calculation the centroid using the coordinates
                # that are available in color_array
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[0]
                    centroid2 = colors_array[6]
                    point1 = (centroid1[5]+(centroid1[7]//2), centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2), centroid2[6]+(centroid2[8]//2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

# method to roatate left face in counter clock wise direction
def left_face_counter_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: L CounterClockwise")
    temp = np.copy(front_face)
    front_face[0, 0] = down_face[0, 0]
    front_face[0, 3] = down_face[0, 3]
    front_face[0, 6] = down_face[0, 6]
    down_face[0, 0] = back_face[0, 8]
    down_face[0, 3] = back_face[0, 5]
    down_face[0, 6] = back_face[0, 2]
    back_face[0, 2] = up_face[0, 6]
    back_face[0, 5] = up_face[0, 3]
    back_face[0, 8] = up_face[0, 0]
    up_face[0, 0] = temp[0, 0]
    up_face[0, 3] = temp[0, 3]
    up_face[0, 6] = temp[0, 6]
    left_face = rotate_counter_clock_wise(left_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # if detected face and actual face after the rotation is same then return the updated faces
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face

                # else display the arrow by calculation the centroid using the coordinates
                # that are available in color_array
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[6]
                    centroid2 = colors_array[0]
                    point1 = (centroid1[5]+(centroid1[7]//2), centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2), centroid2[6]+(centroid2[8]//2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

# method to rotate front face in clock wise direction
def front_face_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print(front_face)
    print("Next Move: F Clockwise")
    temp1 = np.copy(front_face)
    temp = np.copy(up_face)
    front_face = rotate_clock_wise(front_face)
    temp2 = np.copy(front_face)

    # checking the condition and applying neccesary methods on each face and returning the updated faces
    if np.array_equal(temp2, temp1) == True:
        [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
        [up_face, right_face, front_face, down_face, left_face, back_face] = left_face_clock_wise(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
        [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
        return up_face, right_face, front_face, down_face, left_face, back_face
    up_face[0, 8] = left_face[0, 2]
    up_face[0, 7] = left_face[0, 5]
    up_face[0, 6] = left_face[0, 8]
    left_face[0, 2] = down_face[0, 0]
    left_face[0, 5] = down_face[0, 1]
    left_face[0, 8] = down_face[0, 2]
    down_face[0, 2] = right_face[0, 0]
    down_face[0, 1] = right_face[0, 3]
    down_face[0, 0] = right_face[0, 6]
    right_face[0, 0] = temp[0, 6]
    right_face[0, 3] = temp[0, 7]
    right_face[0, 6] = temp[0, 8]

    #front_face = temp

    print(front_face)
    faces = []
    while True:

        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # check for current face and actual face that has to be there after the move is made
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face

                # else display the arrow by calculation the centroid using the coordinates
                # that are available in color_array
                elif np.array_equal(detected_face,temp1) == True:
                    centroid1 = colors_array[8]
                    centroid2 = colors_array[6]
                    centroid3 = colors_array[0]
                    centroid4 = colors_array[2]
                    point1 = (centroid1[5] + (centroid1[7] // 4), centroid1[6] + (centroid1[7] // 2))
                    point2 = (centroid2[5] + (3 * centroid2[8] // 4), centroid2[6] + (centroid2[8] // 2))
                    point3 = (centroid2[5] + (centroid2[7] // 2), centroid2[6] + (centroid2[7] // 4))
                    point4 = (centroid3[5] + (centroid3[8] // 2), centroid3[6] + (3 * centroid3[8] // 4))
                    point5 = (centroid3[5] + (3 * centroid3[8] // 4), centroid3[6] + (centroid3[8] // 2))
                    point6 = (centroid4[5] + (centroid4[8] // 4), centroid4[6] + (centroid4[8] // 2))
                    point7 = (centroid4[5] + (centroid4[8] // 2), centroid4[6] + (3 * centroid4[8] // 4))
                    point8 = (centroid1[5] + (centroid1[8] // 2), centroid1[6] + (centroid1[8] // 4))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

# method to move front face in counter clock wise direction
def front_face_counter_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: F CounterClockwise")
    temp = np.copy(up_face)
    temp1 = np.copy(front_face)
    front_face = rotate_counter_clock_wise(front_face)
    temp2 = np.copy(front_face)
    if np.array_equal(temp2,temp1) == True:
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = left_face_counter_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face)
            return up_face,right_face,front_face,down_face,left_face,back_face
    up_face[0, 6] = right_face[0, 0]
    up_face[0, 7] = right_face[0, 3]
    up_face[0, 8] = right_face[0, 6]
    right_face[0, 0] = down_face[0, 2]
    right_face[0, 3] = down_face[0, 1]
    right_face[0, 6] = down_face[0, 0]
    down_face[0, 0] = left_face[0, 2]
    down_face[0, 1] = left_face[0, 5]
    down_face[0, 2] = left_face[0, 8]
    left_face[0, 8] = temp[0, 6]
    left_face[0, 5] = temp[0, 7]
    left_face[0, 2] = temp[0, 8]

    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp1) == True:
                    centroid1 = colors_array[2]
                    centroid2 = colors_array[0]
                    centroid3 = colors_array[6]
                    centroid4 = colors_array[8]
                    point1 = (centroid1[5] + (centroid1[7] // 4), centroid1[6] + (centroid1[7] // 2))
                    point2 = (centroid2[5] + (3 * centroid2[8]//4), centroid2[6] + (centroid2[8] // 2))
                    point3 = (centroid2[5] + (centroid2[7] // 2), centroid2[6] + (3 * centroid2[7] // 4))
                    point4 = (centroid3[5] + (centroid3[8] // 2), centroid3[6] + (centroid3[8] // 4))
                    point5 = (centroid3[5] + (3 * centroid3[8] // 4), centroid3[6] + (centroid3[8] // 2))
                    point6 = (centroid4[5] + (centroid4[8] // 4), centroid4[6] + (centroid4[8] // 2))
                    point7 = (centroid4[5] + (centroid4[8] // 2), centroid4[6] + (centroid4[8] // 4))
                    point8 = (centroid1[5] + (centroid1[8] // 2), centroid1[6] + (3 * centroid1[8] // 4))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def back_face_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: B Clockwise")
    temp = np.copy(up_face)
    up_face[0, 0] = right_face[0, 2]
    up_face[0, 1] = right_face[0, 5]
    up_face[0, 2] = right_face[0, 8]
    right_face[0, 8] = down_face[0, 6]
    right_face[0, 5] = down_face[0, 7]
    right_face[0, 2] = down_face[0, 8]
    down_face[0, 6] = left_face[0, 0]
    down_face[0, 7] = left_face[0, 3]
    down_face[0, 8] = left_face[0, 6]
    left_face[0, 0] = temp[0, 2]
    left_face[0, 3] = temp[0, 1]
    left_face[0, 6] = temp[0, 0]
    back_face = rotate_clock_wise(back_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def back_face_counter_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: B CounterClockwise")
    temp = np.copy(up_face)
    up_face[0, 2] = left_face[0, 0]
    up_face[0, 1] = left_face[0, 3]
    up_face[0, 0] = left_face[0, 6]
    left_face[0, 0] = down_face[0, 6]
    left_face[0, 3] = down_face[0, 7]
    left_face[0, 6] = down_face[0, 8]
    down_face[0, 6] = right_face[0, 8]
    down_face[0, 7] = right_face[0, 5]
    down_face[0, 8] = right_face[0, 2]
    right_face[0, 2] = temp[0, 0]
    right_face[0, 5] = temp[0, 1]
    right_face[0, 8] = temp[0, 2]
    back_face = rotate_counter_clock_wise(back_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def up_face_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: U Clockwise")
    temp = np.copy(front_face)
    front_face[0, 0] = right_face[0, 0]
    front_face[0, 1] = right_face[0, 1]
    front_face[0, 2] = right_face[0, 2]
    right_face[0, 0] = back_face[0, 0]
    right_face[0, 1] = back_face[0, 1]
    right_face[0, 2] = back_face[0, 2]
    back_face[0, 0] = left_face[0, 0]
    back_face[0, 1] = left_face[0, 1]
    back_face[0, 2] = left_face[0, 2]
    left_face[0, 0] = temp[0, 0]
    left_face[0, 1] = temp[0, 1]
    left_face[0, 2] = temp[0, 2]
    up_face = rotate_clock_wise(up_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[2]
                    centroid2 = colors_array[0]
                    point1 = (centroid1[5]+(centroid1[7]//2), centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2), centroid2[6]+(centroid2[8]//2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def up_face_counter_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: U CounterClockwise")
    temp = np.copy(front_face)
    front_face[0, 0] = left_face[0, 0]
    front_face[0, 1] = left_face[0, 1]
    front_face[0, 2] = left_face[0, 2]
    left_face[0, 0] = back_face[0, 0]
    left_face[0, 1] = back_face[0, 1]
    left_face[0, 2] = back_face[0, 2]
    back_face[0, 0] = right_face[0, 0]
    back_face[0, 1] = right_face[0, 1]
    back_face[0, 2] = right_face[0, 2]
    right_face[0, 0] = temp[0, 0]
    right_face[0, 1] = temp[0, 1]
    right_face[0, 2] = temp[0, 2]
    up_face = rotate_counter_clock_wise(up_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[0]
                    centroid2 = colors_array[2]
                    point1 = (centroid1[5]+(centroid1[7]//2), centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2), centroid2[6]+(centroid2[8]//2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def down_face_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: D Clockwise")
    temp = np.copy(front_face)
    front_face[0, 6] = left_face[0, 6]
    front_face[0, 7] = left_face[0, 7]
    front_face[0, 8] = left_face[0, 8]
    left_face[0, 6] = back_face[0, 6]
    left_face[0, 7] = back_face[0, 7]
    left_face[0, 8] = back_face[0, 8]
    back_face[0, 6] = right_face[0, 6]
    back_face[0, 7] = right_face[0, 7]
    back_face[0, 8] = right_face[0, 8]
    right_face[0, 6] = temp[0, 6]
    right_face[0, 7] = temp[0, 7]
    right_face[0, 8] = temp[0, 8]
    down_face = rotate_clock_wise(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[6]
                    centroid2 = colors_array[8]
                    point1 = (centroid1[5]+(centroid1[7]//2), centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2), centroid2[6]+(centroid2[8]//2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def down_face_counter_clock_wise(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: D CounterClockwise")
    temp = np.copy(front_face)
    front_face[0, 6] = right_face[0, 6]
    front_face[0, 7] = right_face[0, 7]
    front_face[0, 8] = right_face[0, 8]
    right_face[0, 6] = back_face[0, 6]
    right_face[0, 7] = back_face[0, 7]
    right_face[0, 8] = back_face[0, 8]
    back_face[0, 6] = left_face[0, 6]
    back_face[0, 7] = left_face[0, 7]
    back_face[0, 8] = left_face[0, 8]
    left_face[0, 6] = temp[0, 6]
    left_face[0, 7] = temp[0, 7]
    left_face[0, 8] = temp[0, 8]
    down_face = rotate_counter_clock_wise(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[8]
                    centroid2 = colors_array[6]
                    point1 = (centroid1[5]+(centroid1[7]//2), centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2), centroid2[6]+(centroid2[8]//2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def turn_to_right(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: Show Right Face")
    temp = np.copy(front_face)
    front_face = np.copy(right_face)
    right_face = np.copy(back_face)
    back_face = np.copy(left_face)
    left_face = np.copy(temp)
    up_face = rotate_clock_wise(up_face)
    down_face = rotate_counter_clock_wise(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[8]
                    centroid2 = colors_array[6]
                    centroid3 = colors_array[5]
                    centroid4 = colors_array[3]
                    centroid5 = colors_array[2]
                    centroid6 = colors_array[0]
                    point1 = (centroid1[5] + (centroid1[7] // 2), centroid1[6] + (centroid1[7] // 2))
                    point2 = (centroid2[5] + (centroid2[8] // 2), centroid2[6] + (centroid2[8] // 2))
                    point3 = (centroid3[5] + (centroid3[7] // 2), centroid3[6] + (centroid3[7] // 2))
                    point4 = (centroid4[5] + (centroid4[8] // 2), centroid4[6] + (centroid4[8] // 2))
                    point5 = (centroid5[5] + (centroid5[7] // 2), centroid5[6] + (centroid5[7] // 2))
                    point6 = (centroid6[5] + (centroid6[8] // 2), centroid6[6] + (centroid6[8] // 2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def turn_to_front(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: Show Front Face")
    temp = np.copy(front_face)
    front_face = np.copy(left_face)
    left_face = np.copy(back_face)
    back_face = np.copy(right_face)
    right_face = np.copy(temp)
    up_face = rotate_counter_clock_wise(up_face)
    down_face = rotate_clock_wise(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = colors_array[6]
                    centroid2 = colors_array[8]
                    centroid3 = colors_array[3]
                    centroid4 = colors_array[5]
                    centroid5 = colors_array[0]
                    centroid6 = colors_array[2]
                    point1 = (centroid1[5] + (centroid1[7] // 2), centroid1[6] + (centroid1[7] // 2))
                    point2 = (centroid2[5] + (centroid2[8] // 2), centroid2[6] + (centroid2[8] // 2))
                    point3 = (centroid3[5] + (centroid3[7] // 2), centroid3[6] + (centroid3[7] // 2))
                    point4 = (centroid4[5] + (centroid4[8] // 2), centroid4[6] + (centroid4[8] // 2))
                    point5 = (centroid5[5] + (centroid5[7] // 2), centroid5[6] + (centroid5[7] // 2))
                    point6 = (centroid6[5] + (centroid6[8] // 2), centroid6[6] + (centroid6[8] // 2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break




def find_face_in_cube(video, videoWriter, uf, rf, ff, df, lf, bf, text=""):
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        # assinging values to face and blob colors based on the face_detection_in_cube method
        face, colors_array = face_detection_in_cube(bgr_image_input)
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
                # print(np.array_equal(detected_face, tf))
                # print(np.array_equal(detected_face, ff))
                faces = []
                if np.array_equal(detected_face, uf) == False and np.array_equal(detected_face,
                                                                                 ff) == False and np.array_equal(
                        detected_face, bf) == False and np.array_equal(detected_face, df) == False and np.array_equal(
                        detected_face, lf) == False and np.array_equal(detected_face, rf) == False:
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

    # video = cv2.VideoCapture('http://192.168.43.1:8080/video')
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
            # print("Show Front Face")
            front_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                   text="Show Front Face")
            mf = front_face[0, 4]
            print(front_face)
            print(type(front_face))
            print(mf)
            # print("Show Up Face")
            # time.sleep(2)
            up_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                text="Show Top Face")
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
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Down Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                  2, (0, 0, 255), 3)
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
            # print("Show Down Face")
            # time.sleep(2)
            down_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                  text="Show Down Face")
            start_time = datetime.now()
            while True:

                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Right Face", (50, 50),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
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
            # print("Show Right Face")
            # time.sleep(2)
            right_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                   text="Show Right Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Left Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                  2, (0, 0, 255), 3)
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
            # print("Show Left Face")
            # time.sleep(2)
            left_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                  text="Show Left Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Back Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                  2, (0, 0, 255), 3)
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
            # print("Show Back Face")
            # time.sleep(2)
            back_face = find_face_in_cube(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face,
                                  text="Show Back Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    is_ok, bgr_image_input = video.read()
                    if not is_ok:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Front Face", (50, 50),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
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
            # time.sleep(2)
            print(mb)

            # append all the faces in the order so that it can be given to kociemba module
            solution = face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face)
            # print(solution)
            cube_solved = [mu, mu, mu, mu, mu, mu, mu, mu, mu, mr, mr, mr, mr, mr, mr, mr, mr, mr, mf, mf, mf, mf, mf,
                           mf, mf, mf, mf, md, md, md, md, md, md, md, md, md, ml, ml, ml, ml, ml, ml, ml, ml, ml, mb,
                           mb, mb, mb, mb, mb, mb, mb, mb]
            if (face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face) == cube_solved).all():
                # print("CUBE IS SOLVED")
                is_ok, bgr_image_input = video.read()
                bgr_image_input = cv2.putText(bgr_image_input, "CUBE ALREADY SOLVED", (100, 50),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
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
            final_string = ''
            for val in range(len(solution)):
                if solution[val] == mf:
                    final_string = final_string + 'F'
                elif solution[val] == mr:
                    final_string = final_string + 'R'
                elif solution[val] == mb:
                    final_string = final_string + 'B'
                elif solution[val] == ml:
                    final_string = final_string + 'L'
                elif solution[val] == mu:
                    final_string = final_string + 'U'
                elif solution[val] == md:
                    final_string = final_string + 'D'

            print(final_stringinging)
            try:
                solved = kociemba.solve(final_stringing)
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
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_face_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "R'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_counter_clock_wise(video, videoWriter,
                                                                                               up_face, right_face,
                                                                                               front_face, down_face,
                                                                                               left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "R2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_face_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_face_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "L":
                [up_face, right_face, front_face, down_face, left_face, back_face] = left_face_clock_wise(video, videoWriter,
                                                                                             up_face, right_face,
                                                                                             front_face, down_face,
                                                                                             left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "L'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = left_face_counter_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "L2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = left_face_clock_wise(video, videoWriter,
                                                                                             up_face, right_face,
                                                                                             front_face, down_face,
                                                                                             left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = left_face_clock_wise(video, videoWriter,
                                                                                             up_face, right_face,
                                                                                             front_face, down_face,
                                                                                             left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "F":
                [up_face, right_face, front_face, down_face, left_face, back_face] = front_face_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "F'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = front_face_counter_clock_wise(video, videoWriter,
                                                                                               up_face, right_face,
                                                                                               front_face, down_face,
                                                                                               left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "F2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = front_face_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = front_face_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "B":
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video, videoWriter,
                                                                                                   up_face, right_face,
                                                                                                   front_face,
                                                                                                   down_face, left_face,
                                                                                                   back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_face_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video, videoWriter,
                                                                                                   up_face, right_face,
                                                                                                   front_face,
                                                                                                   down_face, left_face,
                                                                                                   back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "B'":
                # print(up_face, right_face, front_face, down_face, left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video, videoWriter,
                                                                                                   up_face, right_face,
                                                                                                   front_face,
                                                                                                   down_face, left_face,
                                                                                                   back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_counter_clock_wise(video, videoWriter,
                                                                                               up_face, right_face,
                                                                                               front_face, down_face,
                                                                                               left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video, videoWriter,
                                                                                                   up_face, right_face,
                                                                                                   front_face,
                                                                                                   down_face, left_face,
                                                                                                   back_face)
            elif step == "B2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video, videoWriter,
                                                                                                   up_face, right_face,
                                                                                                   front_face,
                                                                                                   down_face, left_face,
                                                                                                   back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_face_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = right_face_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video, videoWriter,
                                                                                                   up_face, right_face,
                                                                                                   front_face,
                                                                                                   down_face, left_face,
                                                                                                   back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "U":
                [up_face, right_face, front_face, down_face, left_face, back_face] = up_face_clock_wise(video, videoWriter, up_face,
                                                                                           right_face, front_face,
                                                                                           down_face, left_face,
                                                                                           back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "U'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = up_face_counter_clock_wise(video, videoWriter, up_face,
                                                                                            right_face, front_face,
                                                                                            down_face, left_face,
                                                                                            back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "U2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = up_face_clock_wise(video, videoWriter, up_face,
                                                                                           right_face, front_face,
                                                                                           down_face, left_face,
                                                                                           back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = up_face_clock_wise(video, videoWriter, up_face,
                                                                                           right_face, front_face,
                                                                                           down_face, left_face,
                                                                                           back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "D":
                [up_face, right_face, front_face, down_face, left_face, back_face] = down_face_clock_wise(video, videoWriter,
                                                                                             up_face, right_face,
                                                                                             front_face, down_face,
                                                                                             left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "D'":
                [up_face, right_face, front_face, down_face, left_face, back_face] = down_face_counter_clock_wise(video, videoWriter,
                                                                                              up_face, right_face,
                                                                                              front_face, down_face,
                                                                                              left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))
            elif step == "D2":
                [up_face, right_face, front_face, down_face, left_face, back_face] = down_face_clock_wise(video, videoWriter,
                                                                                             up_face, right_face,
                                                                                             front_face, down_face,
                                                                                             left_face, back_face)
                [up_face, right_face, front_face, down_face, left_face, back_face] = down_face_clock_wise(video, videoWriter,
                                                                                             up_face, right_face,
                                                                                             front_face, down_face,
                                                                                             left_face, back_face)
                # print(face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face))

        # after solving all we just need to check the strings and display message accordingly
        cube_solved = [mu, mu, mu, mu, mu, mu, mu, mu, mu, mr, mr, mr, mr, mr, mr, mr, mr, mr, mf, mf, mf, mf, mf, mf,
                       mf, mf, mf, md, md, md, md, md, md, md, md, md, ml, ml, ml, ml, ml, ml, ml, ml, ml, mb, mb, mb,
                       mb, mb, mb, mb, mb, mb]
        if (face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face) == cube_solved).all():
            # print("CUBE IS SOLVED")
            is_ok, bgr_image_input = video.read()
            bgr_image_input = cv2.putText(bgr_image_input, "CUBE SOLVED", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                          (0, 0, 255), 3)

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
                    bgr_image_input = cv2.putText(bgr_image_input, "CUBE SOLVED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                  (0, 0, 255), 3)
                    videoWriter.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            break
        # print(front_face)
        # print(up_face)

        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        # print(count)
        # print(color_array)
        # print(face)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


if __name__ == "__main__":
    main()
