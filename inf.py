import os
import math
from ultralytics import YOLO
import cv2
import cvzone
import time


print('''
Functions:
    1- img_inf(img_path = None, species = False, resize = True)
    2- vid_inf(video_path = None, output_path = None)
    
Working:
    1- img_inf(img_path = None, species = False, resize = True):
        * If img_path is provided, inference on provided image.
        * If img_path isn't provided, inference on test image.
        * If species is set to True, does inference on species, otherwise, classify as 'Bird'.
        * If resize is set to False, it will not resize the output image.
    
    2- vid_inf(video_path = None, output_path = None):
        * If video_path is provided, does inference on provided video.
        * If output_path is provided, saves the output video on provided path.
        * If no argument is given, does inference on test video.    
        
    3- If python file is run separately, it will do inference on test video first and than the test image
''')


def img_inf(img_path = None, species = False, resize = True):
    if species == True:
        model_path = '/home/msc1/Desktop/YOLO/species_model/train/weights/best.pt'
        cls_name = ['Greatcormorant','GreatwhitePelican','Caspiantern','Gull','Royaltern','Unknown'] 
    elif species == False:
        model_path = "TUNED_lr0_0.01_lrf_0.001_warmup_epochs_5.0_warmup_momentum_0.95_optimizer_auto/train/weights/best.pt"
        cls_name = ['Bird'] 
    if img_path == None:
        img_path = '/home/msc1/Downloads/newmexico/BDA_12C_20181127_1_21.png'
    img = cv2.imread(img_path)
    print(model_path)
    model = YOLO(model_path)  # load model
    threshold = 0.2 # Score threshold
    
    results = model(img ,show = True, stream = True)

    while True:
        try:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    score = math.ceil((box.conf[0]*100)) / 100
                    if score > threshold:
                        #Draw B-Box on object:
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 100), 2)
                        
                        #Write name of class on top:
                        
                        text_size, _ = cv2.getTextSize(f'{cls_name[int(box.cls)]}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        text_w, text_h = text_size
                        cv2.rectangle(img, (int(x1), int(y1 - 30)), (int(x1) + text_w, int(y1)), (0, 0, 100), -1)
                        
                        
                        
                        cv2.putText(img, cls_name[int(box.cls)], (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1, cv2.LINE_AA)
            if resize == True:            
                cv2.imshow("Image Inference",cv2.resize(img,(600,int((600 / img.shape[0]) * img.shape[1]))))  # Display image
            elif resize == False:
                cv2.imshow("Image Inference",img)
            if cv2.waitKey(1) == 27:  # Press escape key to close the window
                break
        except:
            print('No detections')
    cv2.destroyAllWindows()
    

def vid_inf(video_path = None, output_path = None):
    
    if video_path == None:
        video_path = 'test/test.mp4'

    cap = cv2.VideoCapture(video_path)
    # Get video frame dimensions and create a VideoWriter object for saving the output
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if output_path != None:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    model_path = "TUNED_lr0_0.01_lrf_0.001_warmup_epochs_5.0_warmup_momentum_0.95_optimizer_auto/train/weights/best.pt"

    # Load a model
    model = YOLO(model_path)  # load a custom model

    threshold = 0.2
    total_counts = 0

    while True:
        success, img = cap.read()
        
        if not success:
            print("End of video.")
            break
            
        results = model.track(img, stream=True, show=False, persist = True)
            
        for result in results:
                
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx,cy = x1+ w//2, y1+ h//2
                if box.id != None:
                    ID = box.id.cpu().numpy().astype(int)[0]
        
                    score = math.ceil((box.conf[0]*100)) / 100
                    if score > threshold:
                        if cy <=502:
                            cv2.circle(img,(cx, cy), 5, (0,0,255),-1)
                        else:
                            cv2.circle(img,(cx, cy), 5, (0,255,0),-1)
                                
                        if 0 < cx < img.shape[1] and 515 > cy > 485:
                            cv2.circle(img,(cx, cy), 5, (0,0,0),-1)
                            total_counts+= 1

            imgGraphic = cv2.imread('graphic1.png', cv2.IMREAD_UNCHANGED)
            img = cvzone.overlayPNG(img, imgGraphic, (0,0))
            cv2.putText(img, f' {total_counts}', (35, 80),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (0, 500), (img.shape[1], 500), (0, 0, 100), 2)
        if output_path != None:
            out.write(img)
        cv2.imshow("Video Detection", img)
        if cv2.waitKey(0) == 27:                ## Press escape key to come out of the loop (close the camera window)
            break

    cap.release()
    if output_path != None:
        out.release()
    cv2.destroyAllWindows()
        
def main():
    vid_inf()
    img_inf()

if __name__ == "__main__":
    main()
