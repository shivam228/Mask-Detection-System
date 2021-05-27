import cv2
import YOLO
import Detect
capture= cv2.VideoCapture(0)

mask = "No Mask"
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 0, 0) 
thickness = 1
if capture.isOpened() is False:
	print("Error opening camera")
while True:
	ret,frame=capture.read()
	if ret is True:
		detected=YOLO.humanDetect(frame)

		if detected is not None:
	    	
			#cv2.imshow("human detected",detected)
			#cv2.waitKey()
			print("Human Detected")
			face = Detect.detect_face(detected)
			mask = Detect.detect_mask(face)
			print(mask)

		else:
			print('Human Not detected')
		frame = cv2.putText(frame, mask, org, font, fontScale, color, thickness, cv2.LINE_AA) 
		cv2.imshow("Video", frame)
		if cv2.waitKey(2) & 0xFF == ord('q'): 
			break
	

capture.release()
cv2.destroyAllWindows()
