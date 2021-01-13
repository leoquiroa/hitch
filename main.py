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
import math

####################################################
# utils: list
####################################################

def video_list(root):
    file_list = os.listdir(os.path.join(root,'videos','raw'))
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

####################################################
# conversion from original format to mp4
####################################################

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

####################################################
# frame extraction
####################################################

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

####################################################
# facial detection
####################################################

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

####################################################
# duration of the video, duration = frames/fps
####################################################

def video_length(root):
    base_mp4 = os.path.join(root,'videos','mp4')
    file_list = os.listdir(base_mp4)
    j_list = {}
    for f in file_list:
        #print(f)
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

####################################################
# create csv
####################################################

def list_faces_frame_sec(root,n):
    j_videos = video_list(root)
    j_frames = dir_list(root,j_videos,'frames')
    j_duration = video_length(root)
    j_faces = dir_list(root,j_videos,'faces')
    for subject,val_subject in j_frames.items():
        for question,val_question in val_subject.items():
            print(subject,question)
            parameter = [subject,question,val_question]
            [fps,sample] = get_video_specs(j_duration,parameter,n)
            all_crop = crops(root,parameter,j_faces)
            write_csv(subject,question,val_question,fps,sample,all_crop)

def crops(root,parameter,j_faces):
    subject = parameter[0]
    question = parameter[1]
    val_question = parameter[2]
    face_crops = j_faces[subject][question]
    url = os.path.join(root,'faces',subject,question,'crop_')
    positive = get_crop_positive(face_crops,url)
    negative = get_crop_negative(val_question,positive)
    return create_crop_diff(url,positive,negative)

def get_crop_positive(face_crops,url):
    only_crops = [x for x in face_crops if x.startswith(url)]
    crop_tail = [os.path.split(x)[1] for x in only_crops]
    crop_ext = [x.split('_')[1] for x in crop_tail]
    return [int(x[:-4]) for x in crop_ext]

def get_crop_negative(val_question,crop_nums):
    total_frames = len(val_question)
    original = list(range(total_frames))
    return list(set(original) - set(crop_nums))

def create_crop_diff(url,positive,negative):
    positive = [[x,url+str(x)+'.jpg'] for x in positive]
    negative = [[x,'-1'] for x in negative]
    all = negative + positive
    all.sort(key = lambda x:x[0])
    return [x[1] for x in all]

def get_video_specs(j_duration,parameter,n):
    subject = parameter[0]
    question = parameter[1]
    val_question = parameter[2]
    duration = j_duration[subject][question]
    fps = math.floor(len(val_question)/duration)
    sample = n if n == -1 else math.floor(fps/n)
    return [fps,sample]

def write_csv(subject,question,val_question,fps,sample,all_crop):
    sec,part = 0,0
    total_frames = len(val_question)
    for i in range(0,total_frames):
        i_fps = i%fps
        if sample > 0 and (i_fps%sample) == 0: 
            part = n if part >= n else part + 1
        if i_fps == 0: 
            sec += 1
            part = 1
        # all or partition by secs
        if sample == -1:
            final = [subject,question,i,val_question[i],sec,all_crop[i]]
        else:
            final = [subject,question,i,val_question[i],sec,part,all_crop[i]]
        # file
        with open(root+'/faces6.csv', 'a+', newline='\n') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(final)

####################################################
# emotion recognition
####################################################

def emotion_recognition(root,file_name):
    #model
    #model = load_model(os.path.join(root,"model_v6_23.hdf5"))
    #emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
    #label_map = dict((v,k) for k,v in emotion_dict.items()) 
    #csv file
    full_path = os.path.join(root,file_name)
    with open(full_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        csv_content = [x for x in csv_reader]
    #iterate over the list
    check_part,check_sec = 1,1
    check_subject,check_question = '',''
    accum = []
    for content in csv_content:
        # features
        subject = content[0]
        question = content[1]
        sec = content[4]
        part_no = content[5]
        url_crop = content[6]
        #
        if check_subject != subject:
            check_subject = subject
            check_sec = 1
            check_part = 1
            accum = []
            print(subject)
        if check_question != question:
            check_question = question
            check_sec = 1
            check_part = 1
            accum = []
            print(question)
        #
        if check_sec == int(sec) and check_part == int(part_no):
            accum.append(url_crop)
        else:
            z = [x for x in accum if x != "-1"]
            crop = z[0] if len(z)>0 else 'no crop'
            # file
            final = [check_subject,check_question,check_sec,check_part,crop]
            #print(final)
            with open(root+'/steps_'+file_name, 'a+', newline='\n') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(final)
            check_part += 1 
            accum = []
            accum.append(url_crop)
            if int(sec) > check_sec : 
                check_sec += 1
                check_part = 1
        

    '''
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
    '''

####################################################
# main
####################################################

if __name__ == "__main__":
    root = '/home/hitch'
    file_name = 'faces2.csv' 
    emotion_recognition(root,file_name)