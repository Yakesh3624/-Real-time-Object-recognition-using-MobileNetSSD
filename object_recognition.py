import cv2
import numpy as np
import imutils

prototext = "MobileNetSSD_deploy.prototxt.txt"
weight = "MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototext,weight)

classes =["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
#MobileNetSSD is trained only for the above 20 objects

colors = np.random.uniform(0,255,size=(len(classes),3))

cam = cv2.VideoCapture(0)

while True:
    frame = cam.read()[1]
    frame= imutils.resize(frame,width=500)
    h,w = frame.shape[:2]

    #shape has [height,width,channels]
    
    blob_resize = cv2.resize(frame,(300,300))

    blob = cv2.dnn.blobFromImage(blob_resize,0.007843,(300,300),127.5)
    
    #blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    #scalefactor– scale factor basically multiplies(scales) our image channels. And remember that it scales it down by a factor of 1/n, where n is the scalefactor you provided.
    #mean– this is the mean subtracting values. You can input a single value or a 3-item tuple for each channel RGB, it will basically subtract the mean value from all the channels accordingly, this is done to normalize our pixel values. Note: mean argument utilizes the RGB sequence
    
    net.setInput(blob)
    

    detections = net.forward()
    #detections array contains information about the bounding box coordinates, class labels, and confidence scores.
    detshape = detections.shape[2]
    # detections.shape = (1, 1, 100, 7)
    # 1 denotes the number of images passed, 1 may represent some internal organization or structure of the detections, 100 bounding boxes in 300x300 images, 7 denotes class ID, confidence score, and bounding box coordinates

    #[0.  15.  0.9976586  0.06349641  0.29358158  0.9768893  1.0003783] is the 7 values of dimensins[3], it denotes ,class id, confidence, rectangel coordinates(4 values) and finaly last term is extra.
    #detections[1] has batch size
    #detections[2] has number of detected bounding boxes
    #detections[3] has number of elements in each detections(typically 7, including class ID, confidence score, and bounding box coordinates)
    
    for i in np.arange(0,detshape):
        confidence = detections[0,0,i,2] #detections[0,0,i,1] is class name

        if confidence > 0.5:
            class_id = int(detections[0,0,i,1])
            box=detections[0,0,i,3:7]*np.array([frame.shape[0],frame.shape[1],frame.shape[0],frame.shape[1]])
            (x,y,w,h)=box.astype("int")
            cv2.rectangle(frame,(x,y),(w,h),colors[class_id],2)
            cv2.putText(frame,classes[class_id],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[class_id],2)
            
    cv2.imshow("",frame)
    key =cv2.waitKey(1)
    if key == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break
            


        


'''
what is

np.random.uniform
frame.shape[:]
blobFromImage(img,0.007843,(size),127.5)
detctions.shape[2]
np.arange(0,detections)
'''
    
    
