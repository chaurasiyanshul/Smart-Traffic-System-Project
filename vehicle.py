import cv2
import numpy as np



# web camera
vid = cv2.VideoCapture('video.mp4')
min_width_react = 80
min_hight_react = 80

count_line_positon = 550

# Initialize substracter
algo = cv2.createBackgroundSubtractorMOG2()


# for point draw in detected object
def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 6 #allowable error between pixel
counter = 0




while True:
    ret,frame = vid.read()

    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))    #defines shape
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    countershape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25,count_line_positon),(1200, count_line_positon),(255,127,0),3)

    # for rectangle draw in vehicle

    for (i,c) in enumerate(countershape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_react)and(h>= min_hight_react)
        if not validate_counter:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame,center,4, (0,0,255),-1)


        for (x,y) in detect:
            if y<(count_line_positon+offset) and y>(count_line_positon-offset):
                counter+=1
            cv2.line(frame, (25,count_line_positon),(1200, count_line_positon),(0,127,255),3)
            detect.remove((x,y))
            print("Vehicle Counter:"+str(counter))



    cv2.putText(frame,"Vehicle Counter :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)



    #cv2.imshow('detecter',dilatada)
    cv2.imshow('video original',frame)

    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
vid.release()