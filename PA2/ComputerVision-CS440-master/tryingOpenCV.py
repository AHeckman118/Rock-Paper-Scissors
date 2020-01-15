import cv2
import numpy as np 
from collections import deque

## FLAGS ##
demo = True
debug = True
## ##### ##

# OpenCV Variables
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
sensitivity = 75

# Game Variables
states = ['idle', 'start', 'shoot', 'results']
currentState = states[0]
p1Score = 0
cpScore = 0
shakeCount = 0
p1Choice = "Rock"
cpChoice = "Paper"
pResult = "Lose"

# Other Globals
points = deque(maxlen=10) # contains recent positions of tracked objects
dY = 0
#direction = '-'
directionList = deque(maxlen=15)
directionList.appendleft('None')
frameCounter = 15

# Helper function for text
def doubleText(frame, xy, text):
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, text, (xy[0] - 2, xy[1] + 2), font, 1, (0,0,0), 2, cv2.LINE_AA)
	cv2.putText(frame, text, (xy[0], xy[1]), font, 1, (255,255,255), 2, cv2.LINE_AA)

# Functions for image recognition
def feature_detect(img, template):
	#img = cv2.imread(image,0)
	temp = cv2.imread(template,0)
	orb = cv2.ORB_create()

	#locate keypoints
	ikp, des1 = orb.detectAndCompute(img,None)
	tkp, des2 = orb.detectAndCompute(temp,None)
	
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	avg_distance = 0
	for i in range(0,5):
		avg_distance += matches[i].distance
	avg_distance /= 5
	return avg_distance
	
def play_rps(img, mode):
        #compare image to rock, paper, and scissors to see which one it's most like
	rock = feature_detect(img, 'images/rock_template.jpg')
	paper = feature_detect(img, 'images/paper_template.jpg')
	scissors = feature_detect(img, 'images/scissors_template.jpg')
	rps = [rock,paper,scissors]

	#determine output based on current strategy, default set to win.
	if mode == 'win':
		if min(rps) == rock:
			return ('Rock!', 'Paper!')
		elif min(rps) == paper:
			return ('Paper!','Scissors!')
		else:
			return ('Scissors!','Rock!')
	elif mode == 'draw':
		if min(rps) == rock:
			return ('Rock!', 'Rock!')
		elif min(rps) == paper:
			return ('Paper!', 'Paper!')
		else:
			return ('Scissors!', 'Scissors!')
	else:
		if min(rps) == rock:
			return('Rock!', 'Scissors!')
		elif min(rps) == paper:
			return('Paper!', 'Rock!')
		else:
			return('Scissors!','Paper!')

# Loop on frames of video
while True:
	ret, frame = cap.read()
	# frame is unmodified video
	# frame2 is modified 

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#median = cv2.medianBlur(gray, 25)
	blur = cv2.GaussianBlur(gray, (15,15),0)
	no_bg = fgbg.apply(blur)
	#kernel = np.ones((10,10), np.float32)/100
	#smoothed = cv2.filter2D(no_bg, -1, kernel)
	#blur = cv2.GaussianBlur(no_bg, (21,21),0)
	#median = cv2.medianBlur(no_bg, 11)
	#eroded = cv2.erode(no_bg, kernel, iterations = 1)
	#opened = cv2.morphologyEx(no_bg, cv2.MORPH_OPEN, kernel)
	#closed = cv2.morphologyEx(no_bg, cv2.MORPH_CLOSE, kernel)
	#median2 = cv2.medianBlur(opened, 15)
	frame2 = no_bg



	# Game Logic
	if currentState == 'idle':
		# waiting for start gesture
		shakeCount = 0
		p1Choice = "Rock"
		cpChoice = "Paper"
		pResult = "Lose"
		frameCounter = 15

		## movement tracking 

		# finding position of largest detected object, storing their coordinates
		cont_list = cv2.findContours(frame2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		if len(cont_list) > 0:
			c = max(cont_list, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			#cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(frame, (int(x),int(y)), 5, (0, 0, 255), -1)
			points.appendleft((x,y))

		# calculating the "speed" and assigning directions
		if len(points) == 10:
			if points[0] is not None and points[-1] is not None:
				dY = points[0][1] - points[-1][1]
				if np.abs(dY) > sensitivity:
					if np.sign(dY) == -1:
						if directionList[0] != "UP":
							directionList.appendleft("UP")
					elif directionList[0] != "DOWN":
						directionList.appendleft("DOWN")
				elif directionList[0] != "NONE":
					directionList.appendleft("NONE")
					# Motion has just stopped. Lets check if the player has started the game
					dlist = [x for x in directionList if x!="NONE"][:7]
					if dlist == ["DOWN", "UP", "DOWN", "UP", "DOWN", "UP", "DOWN"]:
						currentState = states[1]

	if currentState == 'start':
		# Call function
		roi = frame[320:960, 180:540]
		(p1Choice, cpChoice) = play_rps(roi, 'win')
		currentState = states[2]

	if currentState == 'shoot':
		if frameCounter == 0:
			currentState = states[3]
			frameCounter = 60
			cpScore += 1
		else:
			frameCounter -= 1

	if currentState == 'results':
		if frameCounter == 0:
			currentState = states[0]
		else:
			frameCounter -= 1
		directionList = deque(maxlen=15)
		directionList.appendleft('None')

	# UI For Demo
	if debug:
		doubleText(frame, (30, 650), "Game state: " + currentState)
		doubleText(frame, (30, 200), directionList[0])
		doubleText(frame, (30, 240), str(directionList))


	doubleText(frame, (30,30), "Play Rock Paper Scissors!")
	doubleText(frame, (30,70), "Player Score: " + str(p1Score))
	doubleText(frame, (30,110),"AI Score: "+ str(cpScore))

	if currentState == 'idle':    # 0
		doubleText(frame, (475, 200), "Shake Fist to Begin!")
	if currentState == 'start':   # 1
		doubleText(frame, (600, 200), "Ready")
	if currentState == 'shoot' or currentState == 'results':   # 2 or 3
		doubleText(frame, (500, 400), "AI Choice: " + cpChoice)
	if currentState == 'results': # 3
		doubleText(frame, (550, 200), "You " + pResult + "!")
		doubleText(frame, (500, 440), "Player Choice: " + p1Choice)

	# Display the video
	if demo:
		cv2.imshow('frame', frame)
	else:
		cv2.imshow('frame2', frame2)

	# Close by pressing 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Closes program
cap.release()
cv2.destroyAllWindows()
