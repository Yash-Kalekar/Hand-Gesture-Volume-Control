import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import collections

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

history_length = 5
lm_history = collections.deque(maxlen=history_length)

def low_pass_filter(lmList, history):
    if len(history) < history_length:
        history.append(lmList)
        return lmList
    else:
        history.append(lmList)
        avg_lmList = np.mean(np.array(history), axis=0).tolist()
        return avg_lmList

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:

  while cam.isOpened():
    success, image = cam.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lmList = []
    if results.multi_hand_landmarks:
      myHand = results.multi_hand_landmarks[0]
      for id, lm in enumerate(myHand.landmark):
        h, w, c = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])

    if len(lmList) != 0:
      lmList = low_pass_filter(lmList, lm_history)

      x1, y1 = int(lmList[4][1]), int(lmList[4][2])
      x2, y2 = int(lmList[8][1]), int(lmList[8][2])

      cv2.circle(image, (x1, y1), 15, (255, 255, 255), -1)
      cv2.circle(image, (x2, y2), 15, (255, 255, 255), -1)
      cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
      length = math.hypot(x2 - x1, y2 - y1)

      if length < 50:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

      vol = np.interp(length, [50, 220], [minVol, maxVol])
      volume.SetMasterVolumeLevel(vol, None)
      volBar = np.interp(length, [50, 220], [400, 150])
      volPer = np.interp(length, [50, 220], [0, 100])

      cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
      cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
      cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow('HandDetector', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cam.release()
cv2.destroyAllWindows()