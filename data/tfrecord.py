import tensorflow as tf    
import argparse
from tqdm import tqdm
import glob
import os
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str,default="imdb",
                        help="dataset; wiki or imdb")
    args = parser.parse_args()
    return args
def birth(path):
    img_name = path
    Split=img_name.split("_")
    Birth=Split[-2].split("-")
    Y_birth=int(Birth[0])
    M_birth=int(Birth[1])
    Photo=Split[-1].split(".")
    Photo=int(Photo[0])
    if M_birth>7:
        age=Photo-Y_birth-1
    else:
        age=Photo-Y_birth
    return age
def cv2bytes(image):
    return np.array(cv2.imencode('.png',image)[1]).tobytes()
def mtdetect(img):
    global detector
    try:
        face_list = detector.detect_faces(img) # face detect and alignment
    except:    
        return np.array([-1])
    if face_list==[]:
        return np.array([-1])
    elif np.size(face_list)>1:
        return np.array([-1])
    else:
        for face in face_list:
            box = face["box"]
            box=np.array(box)
            box[box<0]=0
            x,y,w,h = box
            if 6400<h*w<45000:
                return box
            else:
                return np.array([-1])
def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def age_data(image_string,ages,out_20ns,out_4ns,out_1ns):
    feature = {"image":bytes_feature(image_string),
               "age"  :int_feature(ages),
               "age20":int_feature(out_20ns),
               "age4" :int_feature(out_4ns),
               "age1" :int_feature(out_1ns)
               }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString() 
def main():
    args = get_args()
    db = args.db
    cwd=os.getcwd()
    root_path = cwd+"/{}_crop_1".format(db)
    tfrecord_path=cwd+"/{}.tfrecords".format(db)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i in tqdm(sorted(glob.glob(root_path+"/**/*.jpg"))):
            img = cv2.imread(i)
            img = cv2.resize(img,(240,240))
            box= mtdetect(img)
            k=i.split("\\")
            k=k[-1]
            if ~(np.any(box==-1)):
                x,y,w,h = box
                cropped = img[y:y+h, x:x+w]
                del x,y,w,h
            else:
                continue
            image = cv2.resize(cropped, (64, 64))
            image_string = cv2bytes(image)
            #image_string = open(i, 'rb').read() 
            out_ages=int(birth(i))
            out_20ns=int((birth(i))//20*20)
            out_4ns=(int(birth(i))%20//4*4)
            out_1ns=(int(birth(i))%20%4)
            _age_data=age_data(image_string,out_ages,out_20ns,out_4ns,out_1ns)
            writer.write(_age_data) 
            del cropped,box

if __name__ == '__main__':
    detector = MTCNN()
    main()

