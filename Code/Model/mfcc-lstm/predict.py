from keras import backend as k
from keras.models import load_model
import numpy as np
import csv
import os
from OtUtils import read_wave, get_mfcc, enhance_speach

test_data_path = 'D:/Documents/SpeakerRecognition/data/voiceprint-test-b/'

source_file_path = 'D:/Documents/SpeakerRecognition/data/voiceprint-test-b/data/'
enhance_file_path = 'D:/Documents/SpeakerRecognition/data/voiceprint-test-b/enhance/'

file_names = os.listdir(source_file_path)

def enhance():
    step = 0
    for file in file_names:
        input_file_name = source_file_path+file
        output_file_name = enhance_file_path+file
        step += 1
        print(step)
        # print(input_file_name)
        # print(output_file_name)
        enhance_speach.process(input_file_name, output_file_name)

def get_input(filename):
    fs, signal = read_wave.read_wav(enhance_file_path+filename)
    mfcc = get_mfcc.get_feature(fs, signal)
    if len(mfcc) < 80:
        return np.zeros((1, 300, 13))
    elif len(mfcc) < 300:
        # Normalization
        max_mfcc, min_mfcc = mfcc.max(), mfcc.min()
        mfcc = (mfcc - min_mfcc) / (max_mfcc - min_mfcc)
        mfcc.resize((300, 13))
        mfcc = np.reshape(mfcc, (1, 300, 13))
        return mfcc
    elif len(mfcc) > 300:
        mfcc = mfcc[int(mfcc.shape[0]/2-150):int(mfcc.shape[0]/2+150)][:]
        mfcc = np.reshape(mfcc, (1, 300, 13))
        # Normalization
        max_mfcc, min_mfcc = mfcc.max(), mfcc.min()
        mfcc = (mfcc - min_mfcc) / (max_mfcc - min_mfcc)
        return mfcc


def save_result(separator):
    model = load_model('D:/Documents/SpeakerRecognition/Code/Model/mfcc-lstm/lstm-lr0.0001-decay-6-batch60-epoch150.png.h5')

    enrollment_list = []
    with open(test_data_path+'enrollment.csv', 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            enrollment_list.append(row[1])


    test_list = []
    with open(test_data_path+'test.csv', 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            test_list.append(row[0])

    i = 0
    yes_num = 0
    with open('D:/Documents/SpeakerRecognition/Code/Model/mfcc-lstm/res/'+str(separator)+'-Result-B.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["FILE_ID", "IS_FAMILY_MEMBER"]
        writer.writerow(header)
        for test in test_list:
            label = 'NO'
            i = i + 1
            j = 0
            for reg in enrollment_list:
                j = j + 1
                # print(test, reg)
                mfcc_test = get_input(test+'.wav')
                mfcc_reg = get_input(reg+'.wav')
                res = model.predict([mfcc_test, mfcc_reg])
                #print("i, j and res is: ", i, j, res[0][0])
                if(res[0][0] >= separator):
                    print(res[0][0])
                    label = 'YES'
                    yes_num += 1
                    print(i, yes_num)
                    break
            content = [test, label]
            writer.writerow(content)


# method enhance just execute once
# enhance()

save_result(0.501572)


