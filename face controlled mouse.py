from imutils import face_utils
import numpy as np
import pyautogui as ptg
import dlib
import cv2

def er(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def mr(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    D = np.linalg.norm(mouth[12] - mouth[16])
    mar = (A + B + C) / (2 * D)
    return mar
def direc(n_p, a_p, w, h):
    nx, ny = n_p
    x, y = a_p

    if nx > x + 1 * w:
        return 'right'
    elif nx < x - 1 * w:
        return 'left'

    if ny > y + 1 * h:
        return 'down'
    elif ny < y - 1 * h:
        return 'up'

    return '-'

WINK_AR_DIFF_THRESH = 0.06
WINK_AR_CLOSE_THRESH = 0.20

MOUTH_COUNT = 0
EYE_COUNT = 0
WINK_COUNT = 0
INPUT_MODE = False
iclick = False
lwink = False
rwink = False
scrollx = False
ANCHOR_POINT = (0, 0)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)


shape_predictor = "data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

vid = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480

while True:
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    mouth = shape[mStart:mEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    nose = shape[nStart:nEnd]

    temp = leftEye
    leftEye = rightEye
    rightEye = temp

    mar = mr(mouth)
    leftEAR = er(leftEye)
    rightEAR = er(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)
    nose_point = (nose[3, 0], nose[3, 1])
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

    for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
        cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)
        

    if diff_ear > WINK_AR_DIFF_THRESH:

        if leftEAR < rightEAR:
            if leftEAR < 0.12:
                WINK_COUNT += 1

                if WINK_COUNT > 10:
                    ptg.click(button='left')

                    WINK_COUNT = 0

        elif leftEAR > rightEAR:
            if rightEAR < 0.12:
                WINK_COUNT += 1

                if WINK_COUNT >10:
                    ptg.click(button='right')

                    WINK_COUNT = 0
        else:
            WINK_COUNT = 0
    else:
        if ear <= 0.12:
            EYE_COUNT += 1

            if EYE_COUNT > 15:
                scrollx = not scrollx
                EYE_COUNT = 0

        else:
            EYE_COUNT = 0
            WINK_COUNT = 0

    if mar >0.5:
        MOUTH_COUNT += 1

        if MOUTH_COUNT >= 15:
            INPUT_MODE = not INPUT_MODE
            MOUTH_COUNT = 0
            ANCHOR_POINT = nose_point

    else:
        MOUTH_COUNT = 0

    if INPUT_MODE:
        cv2.putText(frame, "FACE-MOUSE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        x, y = ANCHOR_POINT
        nx, ny = nose_point
        w, h = 60, 35
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), RED_COLOR, 2)
        cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

        dir = direc(nose_point, ANCHOR_POINT, w, h)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        drag =15
        if dir == 'right':
            ptg.moveRel(drag, 0)
        elif dir == 'left':
            ptg.moveRel(-drag, 0)
        elif dir == 'up':
            if scrollx:
                ptg.scroll(40)
            else:
                ptg.moveRel(0, -drag)
        elif dir == 'down':
            if scrollx:
                ptg.scroll(-40)
            else:
                ptg.moveRel(0, drag)

    if scrollx:
        cv2.putText(frame, 'SCROLL MODE-ON', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)



    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cv2.destroyAllWindows()
vid.release()
