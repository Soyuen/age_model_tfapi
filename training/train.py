import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras import Model
from aec_model import *
from TYY_utils import mk_dir, load_data_npz
from callbacks import *
import numpy as np
from generators import *
import cv2
import tensorflow as tf
import shutil
import random
import time
import math
from argparse import ArgumentParser
import pandas as pd
def get_args():
    parser = ArgumentParser()

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input file")

    parser.add_argument("--db", type=str, required=True,
                        help="imdb or wiki")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    input_path = args.input
    db_name = args.db
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split
    logging.debug("Loading data...")
    start_decay_epoch = [30,60]
    model = AEC_model((64,64,3))
    save_name = 'aec_model'
    if db_name == "imdb":
        model.compile(optimizer=Adam(), loss={'pred_a':'mae','pre_4':pred4_newloss,'pre_1':pred1_newloss,'pre_cod':cod},loss_weights={'pred_a':3,'pre_4':8.5,'pre_1':12,'pre_cod':18})
        data_size=141583
        train_size=int(0.8*data_size)
        test_size=data_size-train_size
    else:
        model.compile(optimizer=Adam(), loss={'pred_a':'mae','pre_4':wpred4_newloss,'pre_1':wpred1_newloss,'pre_cod':cod},loss_weights={'pred_a':3,'pre_4':8.5,'pre_1':12,'pre_cod':18})
        data_size=38138
        train_size=int(0.8*data_size)
        test_size=data_size-train_size       
        weight_file = "imdb_models/weights.hdf5"
        model.load_weights(weight_file)
        
    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir(db_name+"_models")
    mk_dir(db_name+"_checkpoints")
    
    decaylearningrate = DecayLearningRate(start_decay_epoch)

    callbacks = [ModelCheckpoint(db_name+"_checkpoints/weights.{epoch:02d}-{val_pred_a_loss:.3f}.hdf5",
                                 monitor="val_pred_a_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode="auto"), decaylearningrate
                        ]
    logging.debug("Running training...")
    
    auto=tf.data.experimental.AUTOTUNE
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
      
    dataset = tf.data.Dataset.list_files(input_path)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=auto, num_parallel_calls=auto)
    dataset = dataset.map(read_data, num_parallel_calls=auto)
    #dataset = dataset.shuffle(data_size+1,reshuffle_each_iteration=False)
    train_data=dataset.take(train_size)
    test_data=dataset.skip(train_size)
    train_data = train_data.cache()
    test_data = test_data.cache()
    train_data = train_data.shuffle(train_size,reshuffle_each_iteration=True)
    #test_data = test_data.shuffle(test_size+1,reshuffle_each_iteration=True)
    train_data = train_data.batch(batch_size, drop_remainder=True)
    test_data = test_data.batch(batch_size, drop_remainder=True)  
    train_data = train_data.repeat(nb_epochs)
    test_data = test_data.repeat(nb_epochs)
    train_data = train_data.prefetch(auto) #
    test_data = test_data.prefetch(auto)

    hist = model.fit(train_data,                  
                     steps_per_epoch=(train_size//batch_size),
                     validation_data=test_data,
                     epochs=nb_epochs,
                     verbose=1,
                     validation_steps =(test_size//batch_size),
                     callbacks=callbacks)
    logging.debug("Saving weights...")
    model.save_weights(os.path.join(db_name+"_models", save_name+'.h5'), overwrite=True)
    model.save(os.path.join(db_name+"_models/"+'aec_model.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(db_name+"_models", 'history_'+save_name+'.h5'), "history")
    
    src=db_name+"_checkpoints/"
    dst=db_name+"_models/"
    dir_files=os.listdir(src)
    latest_time=0
    for ffs in dir_files:
        filename=src+ffs
        t=os.stat(filename)
        if t.st_mtime>latest_time:
            latest_time=t.st_mtime
            srcfile=filename
            dstfile=dst+ffs
    shutil.copy(srcfile,dstfile)
    val=srcfile.split("-")[1][:-5]
    dstfile=dst+'weights.hdf5'
    shutil.copy(srcfile,dstfile)


if __name__ == '__main__':
    main()