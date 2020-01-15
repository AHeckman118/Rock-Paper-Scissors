import cv2
import numpy as np
from matplotlib import pyplot as plt
    
def feature_detect(img, template):
    #img = cv2.imread(image,0)
    temp = cv2.imread(template,0)
    orb = cv2.ORB_create()
    
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
    rock = feature_detect(image, 'rock_template.jpg')
    paper = feature_detect(image, 'paper_template.jpg')
    scissors = feature_detect(image, 'scissors_template.jpg')
    rps = [rock,paper,scissors]
    if mode == 'win':
        if min(rps) == rock:
            print('Paper!')
            return
        elif min(rps) == paper:
            print('Scissors!')
            return
        else:
            print('Rock!')
            return
    elif mode == 'draw':
        if min(rps) == rock:
            print('Rock!')
            return
        elif min(rps) == paper:
            print('Paper!')
            return
        else:
            print('Scissors!')
            return
    else:
        if min(rps) == rock:
            print('Scissors!')
            return
        elif min(rps) == paper:
            print('Rock!')
            return
        else:
            print('Paper!')
            return
