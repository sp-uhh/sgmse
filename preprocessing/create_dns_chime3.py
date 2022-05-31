import numpy as np
from glob import glob
from soundfile import read, write
import os
from IPython.display import display, Audio
from soundfile import read, write
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import ensure_dir
import shutil
from os.path import join


def windowing(signal, frame_len, frame_shift):
    m_frames = np.array([signal[start_pos:start_pos + frame_len] for start_pos in
                         range(0, len(signal) - frame_len + 1, frame_shift)])
    return m_frames


##### Settings #####

dns_clean_dir = "/data/DNS-Challenge2020/DNS-Challenge/datasets/clean/"
clean_cut_dir = "/data/DNS-Challenge2020/DNS-Challenge/datasets/clean_cut/"

eps = 1e-14
min_energy = -30.
min_active_speech = 3
speech_break = 0.2
offset_start = 0.5
sr = 16000

######################

rate = 10e-3 # 10 ms
frame_length = int(rate*sr)  #160 samles

###### Cut audio files based on VAD

clean_files = sorted(glob(dns_clean_dir + '/**/*.wav', recursive=True))

for file in tqdm(clean_files):
    x, sr_file = read(file)
    assert sr_file == sr, "Samplerate must be 16Hz!"

    # split input signals into frames
    segments = windowing(x, frame_length, frame_length)

    # simple energy VAD to remove segments not containing speech
    v_speech_energy = np.mean(segments**2, 1)
    v_vad = 10.*np.log10(np.maximum(v_speech_energy, eps) / np.maximum(np.max(v_speech_energy), eps)) > min_energy

    start_phase = 0
    active = 0
    speech_pause = 0
    cuts = []
    i = 0
    intermediate = 0

    speech_start = np.where(v_vad==True)[0][0] 
    #start = int((np.maximum((speech_start - offset_start), 0) * sr))

    # cut off long silence part
    if speech_start > speech_break / rate:
        i = (np.maximum((speech_start - offset_start/rate), 0)).astype(int)
        cuts.append(i) 
    # ensure speech starts with pause
    else:
        while i < len(v_vad) and len(cuts) == 0:  
            if v_vad[i] == False:
                start_phase += 1
                i += 1
                if start_phase >= speech_break / rate:
                    cuts.append(i) 
            else:
                start_phase = 0
                i += 1

    # ensure minimum 3.2 s length
    while i < len(v_vad):   
        if v_vad[i] == True:
            intermediate = 0
            active += 1
            if active >= min_active_speech / rate:  # reached minimum length search for pause
                while i < len(v_vad):
                    if v_vad[i] == False:
                        speech_pause += 1
                        i += 1
                        if speech_pause >= speech_break / rate:
                            cuts.append(i)
                            speech_pause = 0
                            # check for longer breaks
                            while i < len(v_vad):
                                if v_vad[i] == False:
                                    i += 1
                                else:
                                    i = np.maximum(i - int(speech_break / rate), i)
                                    cuts.append(i)
                                    intermediate = 0
                                    break
                            break
                    else:
                        speech_pause = 0
                        i += 1
                active = 0
                i += 1
            else:
                i += 1
        else:
            i += 1
    cuts = (np.array(cuts) * rate * sr).astype(int)

    for i in range(int(len(cuts)/2)):
        x_new = x[cuts[i*2]:cuts[1+2*i]]
        write(clean_cut_dir + file.split("/")[-1][:-4] + f'_{i}.wav', x_new, sr)

### Move speakers in dedicated directories

clean_files = sorted(glob(clean_cut_dir + '/**/*.wav', recursive=True))
file = clean_files[0]

for file in clean_files:
    speaker_dir = file.split("_reader_")[-1].split("_")[0]
    new_path = clean_cut_dir + speaker_dir + "/" + file.split("/")[-1]
    ensure_dir(new_path)
    shutil.move(file, new_path)

### Give new file name based on file size


speaker_dirs = sorted(os.listdir(clean_cut_dir))

for speaker_dir in speaker_dirs:
    dir_path = clean_cut_dir + speaker_dir
    files = glob(dir_path + "/*")
    sizes = [os.path.getsize(file) for file in files]
    indices = np.argsort(sizes)
    for i, index in enumerate(indices):
        file = files[index]
        file_name = file.split("/")[-1]
        target = join(clean_cut_dir, speaker_dir, "{:03d}_".format(i) + file_name)
        os.rename(file, target)


### Get speaker information of used speakers

df = pd.read_csv ('/data/DNS-Challenge2020/DNS-Challenge/datasets/list.csv')

num_speaker_per_gender = 50
num_sp_per_gender_valid = 5
num_sp_per_gender_test = 2

df_female = df.loc[df['gender'] == 'f']
df_male = df.loc[df['gender'] == 'm']

train_female_ids = []
for i in range(num_speaker_per_gender):
    id = df_female.iloc[i]['id']
    id = '{:05d}'.format(int(id))
    train_female_ids.append(id)
    
train_male_ids = []
for i in range(num_speaker_per_gender):
    id = df_male.iloc[i]['id']
    id = '{:05d}'.format(int(id))
    train_male_ids.append(id)

valid_female_ids = []
for i in range(num_speaker_per_gender, num_speaker_per_gender + num_sp_per_gender_valid):
    id = df_female.iloc[i]['id']
    id = '{:05d}'.format(int(id))
    valid_female_ids.append(id)
    
valid_male_ids = []
for i in range(num_speaker_per_gender, num_speaker_per_gender + num_sp_per_gender_valid):
    id = df_male.iloc[i]['id']
    id = '{:05d}'.format(int(id))
    valid_male_ids.append(id)  
       
test_female_ids = []
for i in range(num_speaker_per_gender + num_sp_per_gender_valid, num_speaker_per_gender + num_sp_per_gender_valid + num_sp_per_gender_test):
    id = df_female.iloc[i]['id']
    id = '{:05d}'.format(int(id))
    test_female_ids.append(id)
    
test_male_ids = []
for i in range(num_speaker_per_gender + num_sp_per_gender_valid, num_speaker_per_gender + num_sp_per_gender_valid + num_sp_per_gender_test):
    id = df_male.iloc[i]['id']
    id = '{:05d}'.format(int(id))
    test_male_ids.append(id)  