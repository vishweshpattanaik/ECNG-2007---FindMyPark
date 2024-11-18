import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import yt_dlp
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

#Initalize the firestore database
cred = credentials.Certificate(r"C:\Users\kiara\Downloads\findmypark-758a1-firebase-adminsdk-3erd0-1b3f2456f2.json")
firebase_admin.initialize_app(cred)
db=firestore.client()

model=YOLO('yolov8s.pt')

# Function to get live stream URL from YouTube using yt-dlp
def get_youtube_stream_url(youtube_url):
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']

#Function to display the coordinates of the frame
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

#Function to update the parking spots in the database
def update_parking (docID,stat):
    #Getting a reference to the document to update
    doc_ref=db.collection('parkingSpots').document(docID)
    
    if stat==1:
        doc_ref.update({'isOccupied': True })
    else:
        doc_ref.update({'isOccupied': False})
    
    doc_ref.update({'lastUpdated':firestore.SERVER_TIMESTAMP})
    
    
#Function to update a specific field
def update_db (docID,docField,value):
    #Getting a reference to the document to update
    doc_ref=db.collection('CarParkingLots').document(docID)
    doc_ref.update({docField: value})
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


youtube_url = "https://www.youtube.com/live/qhS14BrYRls"
stream_url = get_youtube_stream_url(youtube_url)

# Capture video stream
cap = cv2.VideoCapture(stream_url)

frame_count = 0  # Initialize frame counter

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

#Declare each parking spot
#Area A
areaA1=[(224,348),(290,379),(329,355),(271,329)]

areaA2=[(329,355),(271,329),(307,319),(366,339)]

areaA3=[(307,319),(366,339),(397,323),(340,304)]

areaA4=[(397,323),(340,304),(370,293),(426,311)]

areaA5=[(370,293),(426,311),(450,298),(402, 281)]

areaA6=[(450,298),(402, 281),(417,271),(471,287)]

areaA7=[(417,271),(471,287),(488,279),(439,265)]

areaA8=[(488,279),(439,265),(459,257),(506,269)]

areaA9=[(459,257),(506,269),(523,262),(476,248)]

areaA10=[(523,262),(476,248),(490,239),(535,253)]

#Area B
areaB1=[(414,385),(495,415),(534,389),(454,355)]

areaB2=[(534,389),(454,355),(477,343),(558,365)]

areaB3=[(558,365),(477,343),(514,323),(579,345)]

areaB4=[(579,345),(514,323),(526,310),(593,327)]

areaB5=[(593,327),(526,310),(542,299),(614,315)]

areaB6=[(614,315),(542,299),(568,284),(626,302)]

areaB7=[(626,302),(568,284),(582,277),(637,290)]

areaB8=[(637,290),(582,277),(595,271),(650,280)]

#Area C
areaC1=[(537,401),(680,399),(679,379),(553,382)]

areaC2=[(679,379),(553,382),(575,354),(681,352)]

areaC3=[(575,354),(681,352),(682,327),(603,325)]

#Area D
areaD1=[(858,389),(970,387),(938,351),(831,363)]

areaD2=[(938,351),(831,363),(802,343),(895,324)]

areaD3=[(802,343),(895,324),(874,310),(774, 323)]

areaD4=[(874,310),(774, 323),(759,311),(840,301)]

areaD5=[(759,311),(840,301),(824,286),(738,293)]

if not cap.isOpened():
    print("Error: Couldn't open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break
        # Skipping roughly every 20s of the livestream
        if frame_count % 900 == 0:
            time.sleep(1)
            frame=cv2.resize(frame,(1020,500))

            results=model.predict(frame)
            #print(results)
            a=results[0].boxes.data
            px=pd.DataFrame(a).astype("float")
            #print(px)
            #area A
            listA1=[]
            listA2=[]
            listA3=[]
            listA4=[]
            listA5=[]
            listA6=[]
            listA7=[]
            listA8=[]
            listA9=[]
            listA10=[]
            #area B
            listB1=[]
            listB2=[]
            listB3=[]
            listB4=[]
            listB5=[]
            listB6=[]
            listB7=[]
            listB8=[]
            #area C
            listC1=[]
            listC2=[]
            listC3=[]
              #area D
            listD1=[]
            listD2=[]
            listD3=[]
            listD4=[]
            listD5=[]
    
            for index,row in px.iterrows():
            #print(row)
 
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[5])
                c=class_list[d]
                if 'car' in c:
                    cx=int(x1+x2)//2
                    cy=int(y1+y2)//2
                    #area A
                    resultsA1=cv2.pointPolygonTest(np.array(areaA1,np.int32),((cx,cy)),False)
                    if resultsA1>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA1.append(c) 
                    resultsA2=cv2.pointPolygonTest(np.array(areaA2,np.int32),((cx,cy)),False)
                    if resultsA2>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA2.append(c)
                    resultsA3=cv2.pointPolygonTest(np.array(areaA3,np.int32),((cx,cy)),False)
                    if resultsA3>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA3.append(c)   
                    resultsA4=cv2.pointPolygonTest(np.array(areaA4,np.int32),((cx,cy)),False)
                    if resultsA4>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA4.append(c)  
                    resultsA5=cv2.pointPolygonTest(np.array(areaA5,np.int32),((cx,cy)),False)
                    if resultsA5>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA5.append(c)  
                    resultsA6=cv2.pointPolygonTest(np.array(areaA6,np.int32),((cx,cy)),False)
                    if resultsA6>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA6.append(c)  
                    resultsA7=cv2.pointPolygonTest(np.array(areaA7,np.int32),((cx,cy)),False)
                    if resultsA7>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA7.append(c)   
                    resultsA8=cv2.pointPolygonTest(np.array(areaA8,np.int32),((cx,cy)),False)
                    if resultsA8>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA8.append(c)
                    resultsA9=cv2.pointPolygonTest(np.array(areaA9,np.int32),((cx,cy)),False)
                    if resultsA9>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA9.append(c)   
                    resultsA10=cv2.pointPolygonTest(np.array(areaA10,np.int32),((cx,cy)),False)
                    if resultsA10>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listA10.append(c)
               
                    #area B
                    resultsB1=cv2.pointPolygonTest(np.array(areaB1,np.int32),((cx,cy)),False)
                    if resultsB1>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listB1.append(c)
                    resultsB2=cv2.pointPolygonTest(np.array(areaB2,np.int32),((cx,cy)),False)
                    if resultsB2>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listB2.append(c)
                    resultsB3=cv2.pointPolygonTest(np.array(areaB3,np.int32),((cx,cy)),False)
                    if resultsB3>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listB3.append(c)   
                    resultsB4=cv2.pointPolygonTest(np.array(areaB4,np.int32),((cx,cy)),False)
                    if resultsB4>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listB4.append(c)  
                    resultsB5=cv2.pointPolygonTest(np.array(areaB5,np.int32),((cx,cy)),False)
                    if resultsB5>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listB5.append(c)  
                    resultsB6=cv2.pointPolygonTest(np.array(areaB6,np.int32),((cx,cy)),False)
                    if resultsB6>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listB6.append(c)  
                    resultsB7=cv2.pointPolygonTest(np.array(areaB7,np.int32),((cx,cy)),False)
                    if resultsB7>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listB7.append(c)   
                    resultsB8=cv2.pointPolygonTest(np.array(areaB8,np.int32),((cx,cy)),False)
                    if resultsB8>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listB8.append(c)
               
                    #area C
                    resultsC1=cv2.pointPolygonTest(np.array(areaC1,np.int32),((cx,cy)),False)
                    if resultsC1>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listC1.append(c)  
                    resultsC2=cv2.pointPolygonTest(np.array(areaC2,np.int32),((cx,cy)),False)
                    if resultsC2>=0:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                        cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                        listC2.append(c)
                    resultsC3=cv2.pointPolygonTest(np.array(areaC3,np.int32),((cx,cy)),False)
                    if resultsC3>=0:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                        cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                        listC3.append(c)
                
                    #area D
                    resultsD1=cv2.pointPolygonTest(np.array(areaD1,np.int32),((cx,cy)),False)
                    if resultsD1>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listD1.append(c)            
                    resultsD2=cv2.pointPolygonTest(np.array(areaD2,np.int32),((cx,cy)),False)
                    if resultsD2>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listD2.append(c)
                    resultsD3=cv2.pointPolygonTest(np.array(areaD3,np.int32),((cx,cy)),False)
                    if resultsD3>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listD3.append(c)   
                    resultsD4=cv2.pointPolygonTest(np.array(areaD4,np.int32),((cx,cy)),False)
                    if resultsD4>=0:
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                       cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                       listD4.append(c)  
                    resultsD5=cv2.pointPolygonTest(np.array(areaD5,np.int32),((cx,cy)),False)
                    if resultsD5>=0:
                      cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                      cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                      listD5.append(c)
        

            #area A
            a1=(len(listA1))
            update_parking('XYVDs8hnlTa80ZDfAsde',a1)
            a2=(len(listA2))
            update_parking('MEsAiJBfGNFByFC9mlBt',a2)
            a3=(len(listA3))
            update_parking('462iqqUqJbFpl2Qnit5L',a3)
            a4=(len(listA4))
            update_parking('LTEGMB573Uc0D4G9uf1G',a4)
            a5=(len(listA5))
            update_parking('ZKZcISffjhrHH53G97An',a5)
            a6=(len(listA6))
            update_parking('UXZbB2YmocJaaLnGDqxm',a6)
            a7=(len(listA7))
            update_parking('bdRn4tfcKRdv8QFPVFeJ',a7)
            a8=(len(listA8))
            update_parking('AkQ1eCGUm6QmjkGKc4Pv',a8)
            a9=(len(listA9))
            update_parking('7Sd3j2aVx9fJ7FwAaC5F',a9)
            a10=(len(listA10))
            update_parking('BCP662Eww4CDd8DCB0qw',a10)
     
            #area B
            b1=(len(listB1))
            update_parking('QBzs7BI3WkHn9EXcb0MC',b1)
            b2=(len(listB2))
            update_parking('l0yeCPGcLwzLYP9hBqpU',b2)
            b3=(len(listB3))
            update_parking('UhsQNV04QAtgMnrMXhC1',b3)
            b4=(len(listB4))
            update_parking('UzYPDjYFphhEZ87Wvw23',b4)
            b5=(len(listB5))
            update_parking('gvRxv2sZ0mGfKpDrkI8o',b5)
            b6=(len(listB6))
            update_parking('BIdu0LK0ctNqxMVg5UyL',b6)
            b7=(len(listB7))
            update_parking('zndl9EL3kSR234vjNR7c',b7)
            b8=(len(listB8))
            update_parking('CvDKUqkCRPVQmqJpjpJM',b8)
    
            #area C
            c1=(len(listC1))
            update_parking('P3MVqIWEjBnURQfoXxXV',c1)
            c2=(len(listC2))
            update_parking('JzBB3Kf2aSqVPQUlNhgT',c2)
            c3=(len(listC3))
            update_parking('mEh2iEp1nV3EzKeDr2CY',c3)
    
            #area D
            d1=(len(listD1))
            update_parking('1i99Lgq76psYbjNa0owq',d1)
            d2=(len(listD2))
            update_parking('COfYVahBxdoU9Yp8ClGp',d2)
            d3=(len(listD3))
            update_parking('2NmXl5aqQuYoxrIDBSge',d3)
            d4=(len(listD4))
            update_parking('7YQWdN5EoIazy99POLzc',d4)
            d5=(len(listD5))
            update_parking('1KjeHzNezaiLcYd5Bagp',d5)
    
            #Calculating the numebr og available spaces
            o=(a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+b1+b2+b3+b4+b5+b6+b7+b8+c2+c3+d2+d3+d4+d5)
            space=(24-o)
            update_db('X1mq2K7aQBQKMR1xXX0N', 'spotsAvailable',space)
            
            #Calculating the number of available handicapped spaces
            h=(c1+d1)
            hSpace=(2-h)
            update_db('X1mq2K7aQBQKMR1xXX0N', 'handicappedSpots',hSpace)
            
    
            #area A
            if a1==1:
                cv2.polylines(frame,[np.array(areaA1,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A1'),(317, 375),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA1,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A1'),(317, 375),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if a2==1:
                cv2.polylines(frame,[np.array(areaA2,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A2'),(350, 360),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA2,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A2'),(350, 360),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if a3==1:
                cv2.polylines(frame,[np.array(areaA3,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A3'),(382, 345),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA3,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A3'),(382, 345),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if a4==1:
                cv2.polylines(frame,[np.array(areaA4,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A4'),(416, 330),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA4,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A4'),(416, 330),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if a5==1:
                cv2.polylines(frame,[np.array(areaA5,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A5'),(440, 315),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA5,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A5'),(440, 315),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if a6==1:
                cv2.polylines(frame,[np.array(areaA6,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A6'),(461, 303),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA6,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A6'),(461, 303),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1) 
            if a7==1:
                cv2.polylines(frame,[np.array(areaA7,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A7'),(478, 295),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA7,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A7'),(478, 295),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if a8==1:
                cv2.polylines(frame,[np.array(areaA8,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A8'),(495, 284),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA8,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A8'),(495, 284),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if a9==1:
                cv2.polylines(frame,[np.array(areaA9,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A9'),(514, 279),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA9,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A9'),(514, 279),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if a10==1:
                cv2.polylines(frame,[np.array(areaA10,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('A10'),(530, 268),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaA10,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('A10'),(530, 268),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    
            #area B
            if b1==1:
                cv2.polylines(frame,[np.array(areaB1,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('B1'),(418, 362),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaB1,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('B1'),(418, 362),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if b2==1:
                cv2.polylines(frame,[np.array(areaB2,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('B2'),(450, 347),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaB2,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('B2'),(450, 347),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if b3==1:
                cv2.polylines(frame,[np.array(areaB3,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('B3'),(473, 331),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaB3,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('B3'),(473, 331),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if b4==1:
                cv2.polylines(frame,[np.array(areaB4,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('B4'),(498, 318),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaB4,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('B4'),(498, 318),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if b5==1:
                cv2.polylines(frame,[np.array(areaB5,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('B5'),(512, 300),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaB5,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('B5'),(512, 300),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if b6==1:
                cv2.polylines(frame,[np.array(areaB6,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('B6'),(531, 291),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaB6,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('B6'),(531, 291),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1) 
            if b7==1:
                cv2.polylines(frame,[np.array(areaB7,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('B7'),(550, 281),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaB7,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('B7'),(550, 281),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if b8==1:
                cv2.polylines(frame,[np.array(areaB8,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('B8'),(579, 270),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaB8,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('B8'),(579, 270),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    
            #area C
            if c1==1:
                cv2.polylines(frame,[np.array(areaC1,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('C1'),(690, 393),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaC1,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('C1'),(690, 393),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if c2==1:
                cv2.polylines(frame,[np.array(areaC2,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('C2'),(694, 373),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaC2,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('C2'),(694, 373),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if c3==1:
                cv2.polylines(frame,[np.array(areaC3,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('C3'),(690, 352),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaC3,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('C3'),(690, 352),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    
            #area D
            if d1==1:
                cv2.polylines(frame,[np.array(areaD1,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('D1'),(820, 383),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaD1,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('D1'),(820, 383),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if d2==1:
                cv2.polylines(frame,[np.array(areaD2,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('D2'),(795, 367),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaD2,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('D2'),(795, 367),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if d3==1:
                cv2.polylines(frame,[np.array(areaD3,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('D3'),(769, 344),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaD3,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('D3'),(769, 344),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if d4==1:
                cv2.polylines(frame,[np.array(areaD4,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('D4'),(744, 328),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaD4,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('D4'),(744, 328),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            if d5==1:
                cv2.polylines(frame,[np.array(areaD5,np.int32)],True,(0,0,255),2)
                cv2.putText(frame,str('D5'),(720, 305),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.polylines(frame,[np.array(areaD5,np.int32)],True,(0,255,0),2)
                cv2.putText(frame,str('D5'),(720, 305),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
   
    
            cv2.putText(frame,str(space),(17, 40),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
            cv2.putText(frame,',',(85, 40),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
            cv2.putText(frame,str(hSpace),(121, 40),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

            cv2.imshow("RGB", frame)
            
        # Increment frame count
        frame_count += 1

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
#stream.stop()




