# coding:utf-8
import os
import os
import numpy as np
import pandas as pd
import math
#train_wav to featuretxt
audio_path='D:/ser/Train/Audio'
output_path='D:/ser/Train/f1582'
audio_list=os.listdir(audio_path)
for audio in audio_list:
    if audio[-4:]=='.wav':
        this_path_input=os.path.join(audio_path,audio)
        this_path_output=os.path.join(output_path,audio[:-4]+'.txt')
        cmd='cd /d D:/opensmile-2.3.0/bin/Win32 && SMILExtract_Release -C D:/opensmile-2.3.0/config/IS09_emotion.conf -I '+this_path_input+' -O '+this_path_output
    os.system(cmd)
audio_path='D:/ser/Test/Audio'
#test_wav to featuretxt
output_path='D:/ser/Test/f1582'
audio_list=os.listdir(audio_path)
for audio in audio_list:
    if audio[-4:]=='.wav':
        this_path_input=os.path.join(audio_path,audio)
        this_path_output=os.path.join(output_path,audio[:-4]+'.txt')
        cmd='cd /d D:/opensmile-2.3.0/bin/Win32 && SMILExtract_Release -C D:/opensmile-2.3.0/config/IS13_ComParE.conf -I '+this_path_input+' -O '+this_path_output
    os.system(cmd)
#train_featuretxt to array
txt_path='D:/ser/Train/f1582'
txt_list=os.listdir(txt_path)
features_list=[]
for txt in txt_list:
    if txt[-4:]=='.txt':
        this_path=os.path.join(txt_path,txt)
        f=open(this_path)
        last_line=f.readlines()[-1]
        f.close()
        features=last_line.split(',')
        features=features[1:-1]
        features_list.append(features)
features_array=np.array(features_list)
np.save('D:/ser/Train/opensmile_features1582.npy',features_array)
#test_featuretxt to array
txt_path='D:/ser/Test/f6373'
txt_list=os.listdir(txt_path)
features_list=[]
for txt in txt_list:
    if txt[-4:]=='.txt':
        this_path=os.path.join(txt_path,txt)
        f=open(this_path)
        last_line=f.readlines()[-1]
        f.close()
        features=last_line.split(',')
        features=features[1:-1]
        features_list.append(features)
features_array=np.array(features_list)
np.save('D:/ser/Test/opensmile_features6373.npy',features_array)
