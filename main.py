import os
import cv2
from pathlib import Path
import numpy as np
from keras.models import load_model
import csv
from csv import reader
import subprocess
from moviepy.editor import VideoFileClip
from os import path

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

def ffmpeg_convert(input_video,output_video):
    cmd = ['ffmpeg','-i',input_video,output_video]
    result = subprocess.run(cmd)

def video_convertion(root):
    base_raw = os.path.join(root,'videos','raw')
    file_list = os.listdir(base_raw)
    j_list = {}
    for f in file_list:
        print(f)
        first = f.split('_')
        user = first[0]
        question = first[2].split('.')[0]
        # conversion
        output_video = os.path.join(root,'videos','mp4',user+'_'+question+'.mp4')
        if not path.exists(output_video):
            input_video = os.path.join(base_raw,f)
            ffmpeg_convert(input_video,output_video)

def video_length(root):
    base_mp4 = os.path.join(root,'videos','mp4')
    file_list = os.listdir(base_mp4)
    j_list = {}
    for f in file_list:
        print(f)
        first = f.split('_')
        user = first[0]
        question = first[1].split('.')[0]
        output_video = os.path.join(base_mp4,f)
        # get values per video
        cap = cv2.VideoCapture(output_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(frame_count/fps,2)
        cap.release()
        #creaate dict
        if user in j_list:
            j_list[user].update({question:duration})
        else:
            j_list[user] = {question:duration}
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
        
def dir_list(root,j_list,folder):
    j_images = {}
    for k_subject,v_subject in j_list.items():
        for k_question,v_question in v_subject.items():
            single_question = os.path.join(root,folder,k_subject,k_question)
            all_images = sorted(Path(single_question).iterdir(), key=os.path.getmtime)
            all_images = [str(x) for x in all_images]
            if k_subject in j_images:
                j_images[k_subject].update({k_question:all_images})
            else:
                j_images[k_subject] = {k_question:all_images}
    return j_images

def facial_detection(j_frames):
    url = os.path.join(root,'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(url)
    for k_subject,v_subject in j_frames.items():
        subject = os.path.join(root,'faces',k_subject)
        print(subject)
        if not os.path.isdir(subject): os.mkdir(subject)
        for k_question,v_question in v_subject.items():
            question = os.path.join(subject,k_question)
            print(question)
            if not os.path.isdir(question): os.mkdir(question)
            for url in v_question:
                # rgb image
                img = cv2.imread(url)
                # gray image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # haar cascade face detector
                faces = face_cascade.detectMultiScale(gray, 
                    scaleFactor=1.05, 
                    minNeighbors=4, 
                    minSize=(30,30))
                # no face condition
                if len(faces) == 0 : continue
                # get the area
                areas = [w*h for x,y,w,h in faces]
                # get the maximum of the areas
                i_biggest = np.argmax(areas)
                # filter only the biggest face
                face = faces[int(i_biggest)].reshape(1,4)
                for (x, y, w, h) in face:
                    # cropped and resized image
                    crop_img = img[y:y+h, x:x+w].copy()
                    resized = cv2.resize(crop_img, (150,150))
                    # draw a rectangke over the image
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # get the original name
                h,tail = os.path.split(url)
                # save the square image
                face_path = os.path.join(question,'square_'+tail)
                cv2.imwrite(face_path, img)
                # save the cropped face
                crop_path = os.path.join(question,'crop_'+tail)
                cv2.imwrite(crop_path, resized)

def emotion_recognition(root,j_frames):
    model = load_model(os.path.join(root,"model_v6_23.hdf5"))
    emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
    label_map = dict((v,k) for k,v in emotion_dict.items()) 
    for k_subject,v_subject in j_frames.items():
        if k_subject == 'P1iGNqnOaCUqtnF' : continue
        for k_question,v_question in v_subject.items():
            print(k_subject,k_question)
            for crop_url in v_question:
                h,tail = os.path.split(crop_url)
                if tail.startswith("square_"): continue
                face_image = cv2.imread(crop_url,0)
                face_image = cv2.resize(face_image, (48,48))
                # required shape [1,x,y,1]
                face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
                mylist = []
                try:
                    predicted_class = np.argmax(model.predict(face_image))
                    predicted_label = label_map[predicted_class]
                    #print(tail,predicted_label)
                    mylist = [k_subject,k_question,tail,predicted_label]
                except KeyError as e:
                    mylist = [k_subject,k_question,tail,'no label']
                    #print(tail,'no label')
                with open(root+'/list.csv', 'a+', newline='\n') as f:
                    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                    wr.writerow(mylist)

def individual_emotion(root,csv_file):
    full_path = os.path.join(root,csv_file)
    with open(full_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        user = None
        question = None
        emotions = []
        ready = False
        for row in csv_reader:
            if user == row[0] and question == row[1]: 
                emotions.append(row[3])
                ready = False
            else:
                user = row[0]
                question = row[1]
                emotions.append(row[3])
                ready = True
            if ready == True and len(emotions)>1:
                print(len(emotions))


if __name__ == "__main__":
    root = '/home/hitch'
    #j_videos = video_list(root)
    #analyze_user(root,j_list) #extract frames
    #j_frames = dir_list(root,j_videos,'frames')
    #facial_detection(j_frames)
    
    # for csv
    #j_videos = video_list(root)
    #j_faces = dir_list(root,j_videos,'faces')
    #emotion_recognition(root,j_faces)

    #j_videos = video_convertion(root)
    j_videos = video_length(root)
    print(j_videos)
    #individual_emotion(root,'list.csv')





    