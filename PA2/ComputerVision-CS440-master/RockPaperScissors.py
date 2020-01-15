import cv2
import numpy as np 
from collections import deque

## FLAGS ##
demo = True
debug = False
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
def feature_detect(img, temp):
	#img = cv2.imread(image,0)
	#temp = cv2.imread(template,0)
        #create ORB object
	orb = cv2.ORB_create()

	#locate keypoints
	ikp, des1 = orb.detectAndCompute(img,None)
	tkp, des2 = orb.detectAndCompute(temp,None)
	
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)

	#compute avg distance between keypoints to find the most similar template
	avg_distance = 0
	for i in range(0,5):
		avg_distance += matches[i].distance
	avg_distance /= 5
	return avg_distance
	
def play_rps(img, mode):
        #compare whited out image to whited out templates
	white = make_template_white()
	rock = feature_detect(img, white[0])
	paper = feature_detect(img, white[1])
	scissors = feature_detect(img, white[2])
	rps = [rock,paper,scissors]

	#throw a hand, depending on strategy
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

def make_template_white():
    rock = cv2.imread("images/rock_template.jpg",0)
    paper = cv2.imread("images/paper_template.jpg",0)
    scissors = cv2.imread("images/scissors_template.jpg",0)
    ret,thresh1 = cv2.threshold(rock,155,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(paper,250,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(scissors,250,255,cv2.THRESH_BINARY_INV)
    titles = ['rock','paper','scissors']
    images = [thresh1,thresh2,thresh3]
    return [thresh1,thresh2,thresh3]

# Loop on frames of video
while True:
	ret, frame = cap.read()
	# frame is unmodified video
	# frame2 is modified 

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (15,15),0)
	no_bg = fgbg.apply(blur)
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
		#roi = frame[320:960, 180:540]
		(p1Choice, cpChoice) = play_rps(frame2, 'win')
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
		doubleText(frame, (30, 240), str(directionList))


	doubleText(frame, (30,30), "Play Rock Paper Scissors!")
	doubleText(frame, (30,70), "Player Score: " + str(p1Score))
	doubleText(frame, (30,110),"AI Score: "+ str(cpScore))

	if currentState == 'idle':    # 0
		doubleText(frame, (475, 200), "Shake Fist to Begin!")
		doubleText(frame, (30, 200), directionList[0])
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
