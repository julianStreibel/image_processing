import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_content/output.mp4', fourcc, 7.0, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('frame',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # q to exit
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()