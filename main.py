import os
import cv2
from pathlib import Path

def create_list(root):
    file_list = os.listdir(root)
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
        
def cosa(root,j_list):
    j_stats = {}
    all_files = []
    for key,values in j_list.items():
        user_folder = os.path.join(root,'frames',key)
        question_folder = len(os.listdir(user_folder))+1
        for q in range(1,question_folder):
            single_question = os.path.join(user_folder,str(q))
            all_frames = os.listdir(single_question)
            no = len(all_frames)
            j_stats[key+'.'+str(q)] = no
            z = [os.path.join(single_question,str(f)+'.jpg') for f in range(0,no)]
            all_files.append(z)
    return [j_stats,all_files]

if __name__ == "__main__":
    root = '/home/hitch'
    j_list = create_list(os.path.join(root,'videos'))
    analyze_user(root,j_list)
    #[j_stats,all_files] = cosa(root,j_list)
    #unique_full_list = sum(all_files,[])




    