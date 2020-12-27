import os
import cv2
from pathlib import Path

def video_list(root):
    file_list = os.listdir(os.path.join(root,'videos'))
    j_list = {}
    for f in file_list:
        first = f.split('_')
        user = first[0]
        question = first[2].split('.')[0]
        if user in j_list:
            j_list[user].update({question:f})
        else:
            j_list[user] = {question:f}
    return j_list

def analyze_user(root,j_list):
    #subject cycle
    for key,values in j_list.items():
        print(key)
        #subject folder
        subject = os.path.join(root,'frames',key)
        if not os.path.isdir(subject): os.mkdir(subject)
        #question cycle
        analyze_questions(values,subject,root)

def analyze_questions(values,subject,root):
    for q,url in values.items():
        print(q)
        #question folder
        question = os.path.join(subject,q)
        if not os.path.isdir(question): os.mkdir(question)
        #video path build and read video
        single_video = os.path.join(root,'videos',url)
        get_and_save_frames(single_video,subject,q)

def get_and_save_frames(single_video,subject,q):
    #video path build and read video
    vidcap = cv2.VideoCapture(single_video)
    success,image = vidcap.read()
    count = 0
    # video cycle
    while success:
        frame_path = os.path.join(subject,q,str(count)+'.jpg')
        cv2.imwrite(frame_path, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
        
def frame_list(root,j_list):
    j_frames = {}
    for k_subject,v_subject in j_list.items():
        for k_question,v_question in v_subject.items():
            single_question = os.path.join(root,'frames',k_subject,k_question)
            all_frames = sorted(Path(single_question).iterdir(), key=os.path.getmtime)
            all_frames = [str(x) for x in all_frames]
            if k_subject in j_frames:
                j_frames[k_subject].update({k_question:all_frames})
            else:
                j_frames[k_subject] = {k_question:all_frames}
    return j_frames

def facial_detection(j_frames):
    url = os.path.join(root,'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(url)
    for k_subject,v_subject in j_frames.items():
        subject = os.path.join(root,'faces',k_subject)
        if not os.path.isdir(subject): os.mkdir(subject)
        for k_question,v_question in v_subject.items():
            question = os.path.join(subject,k_question)
            if not os.path.isdir(question): os.mkdir(question)
            for url in v_question:
                img = cv2.imread(url)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 
                    scaleFactor=1.05, 
                    minNeighbors=4, 
                    minSize=(30,30))
                if len(faces) == 0 : continue
                for (x, y, w, h) in faces:
                    crop_img = img[y:y+h, x:x+w].copy()
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                h,tail = os.path.split(url)
                face_path = os.path.join(question,'square_'+tail)
                cv2.imwrite(face_path, img)
                crop_path = os.path.join(question,'crop_'+tail)
                cv2.imwrite(crop_path, crop_img)

if __name__ == "__main__":
    root = '/home/hitch'
    j_videos = video_list(root)
    #analyze_user(root,j_list)
    j_frames = frame_list(root,j_videos)
    facial_detection(j_frames)




    