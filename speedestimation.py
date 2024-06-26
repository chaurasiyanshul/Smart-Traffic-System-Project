import ultralytics
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import* # Assuming this is the correct import
import time


model = YOLO('yolov8n.pt')

class_list  = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic Light','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','Toothbrush']
tracker = Tracker()  # Corrected tracker initialization
count = 0
down = {}
up = {}

counter_down = []
counter_up = []

cap = cv2.VideoCapture('highway.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype('float')
    print(px)

    list = []

    for index,row in px.iterrows():
        print(row)
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
            #print(c)

    bbox_id = tracker.update(list)
    #print(bbox_id)
    for bbox in bbox_id:
        x3, y3, x4,y4,id = bbox
        cx = int(x3+x4)//2  # cx,cy central quardinate of the bounding box
        cy = int(y3+y4)//2

        #cv2.circle(frame,(cx,cy),4,(0,0,255),-1)      #putting dot at the center of the object
        #cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)

        red_line_y = 192
        blue_line_y = 268
        offset = 7



#condition for the red line
        # if red_line_y < (cy + offset) and red_line_y > (cy - offset):
        #     down[id] = time.time()


        #     if id in down:
        #         cv2.circle(frame,(cx,cy), 4 , (0,0,255), -1)
        #         cv2.putText(frame, str(id),(cx,cy), cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255,),2)
        # print(down)
    
#conditon for the blue line
        # if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
        #     up[id] = time.time()


        #     if id in up:
        #         cv2.circle(frame,(cx,cy), 4 , (0,0,255), -1)
        #         cv2.putText(frame, str(id),(cx,cy), cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255,),2)
        # print(up)


        if red_line_y<(cy+offset) and red_line_y > (cy-offset):


            down[id] = time.time() # current time when car touch the first line

        if id in down:
            if blue_line_y<(cy+offset) and blue_line_y>(cy-offset):
                elapsed_time = time.time()- down[id]
                if counter_down.count(id)==0:
                    counter_down.append(id)
                    distance = 200 #meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6 #this will give speed killometers
                    cv2.circle(frame,(cx,cy), 4 , (0,0,255), -1)
                    cv2.rectangle(frame, (x3,y3), (x4,y4), (0,255,0),2)
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    cv2.putText(frame,str(int(a_speed_kh)) + 'km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0))
# going up
        if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
        

            up[id] = time.time() # current time when car touch the first line

        if id in up:
        
            if red_line_y < (cy+offset) and red_line_y>(cy-offset):
                elapsed1_time = time.time()- up[id]
                if counter_up.count(id)==0:
                    counter_up.append(id)
                    distance1 = 10 #meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6 #this will give speed killometers
                    cv2.circle(frame,(cx,cy), 4 , (0,0,255), -1)
                    cv2.rectangle(frame, (x3,y3), (x4,y4), (0,255,0),2)
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    cv2.putText(frame,str(int(a_speed_kh1)) + 'km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0))





    text_color = (255,255,255) #white color for text
    blue_color = (255,0,0) #B,G,R
    green_color = (0,255,0)
    red_color = (0,0,255)


    cv2.line(frame,(172,198),(774,198),red_color,3) 
    cv2.putText(frame,("red  line"),(172,198),cv2.FONT_HERSHEY_COMPLEX,0.5,text_color, 1,cv2.LINE_AA)


    cv2.line(frame,(8,268),(927,268),blue_color,3)
    cv2.putText(frame,('blue line'),(8,268),cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color, 1 , cv2.LINE_AA)


    cv2.putText(frame, f'Going down-{len(counter_down)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f'Going up-{len(counter_up)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)




    cv2.imshow('frames', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Changed waitKey to 1 for smoother video display
        break

cap.release()
cv2.destroyAllWindows()
