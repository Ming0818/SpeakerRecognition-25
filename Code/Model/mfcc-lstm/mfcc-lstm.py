from keras.layers import Input, LSTM, Dense, Flatten, concatenate, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import plot_model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from OtUtils import read_wave, PathConfig
from OtUtils import get_mfcc

import datetime as d

file_path = PathConfig.output_file_path
label_info_path = PathConfig.label_info_path

batch_size = 120
epochs = 150
lr = 0.0001


def get_input_feature(file_name):
    fs, signal = read_wave.read_wav(file_name)
    mfcc = get_mfcc.get_feature(fs, signal)
    if len(mfcc) < 80:
        flag = 1
        mfcc = np.zeros((300, 13))
        return flag, mfcc
    elif len(mfcc) < 300:
        # Normalization
        max_mfcc, min_mfcc = mfcc.max(), mfcc.min()
        mfcc = (mfcc - min_mfcc) / (max_mfcc - min_mfcc)

        flag = 0
        mfcc = np.resize(mfcc, (300, 13))
        return flag, mfcc
    elif len(mfcc) > 300:
        mfcc = mfcc[int(mfcc.shape[0]/2-150):int(mfcc.shape[0]/2+150)][:]
        flag = 0

        # Normalization
        max_mfcc, min_mfcc = mfcc.max(), mfcc.min()
        mfcc = (mfcc - min_mfcc) / (max_mfcc - min_mfcc)
        return flag, mfcc


###############################
# build model construct begin #
###############################


input_one = Input(shape=(300, 13), dtype='float32', name='input_one')
lstm_out_one = LSTM(units=13, activation='tanh', dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(input_one)
lstm_out_one = Flatten()(lstm_out_one)


input_two = Input(shape=(300, 13), dtype='float32', name='input_two')
lstm_out_two = LSTM(units=13, activation='tanh', dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(input_two)
lstm_out_two = Flatten()(lstm_out_two)

x = concatenate([lstm_out_one, lstm_out_two])
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=[input_one, input_two], output=[output])

optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

plot_model(model, to_file='./model-lstm-lr0.0001-decay-6-batch60-epoch150.png')

##############################
#  build model construct end #
##############################

wav_label = pd.read_csv(label_info_path)
wav_label = wav_label.sample(frac=1)
log = []

def generate_batch(mode):
    while 1:
        train_inputs_one = np.zeros((1, 300, 13))
        train_inputs_two = np.zeros((1, 300, 13))
        train_labels = np.array([])

        for i in range(0, len(wav_label)):
            file_one = wav_label.iloc[i][0]
            file_two = wav_label.iloc[i][1]
            train_label = wav_label.iloc[i][2]
            try:
                flag_one, train_input_one = get_input_feature(file_path + file_one + '.wav')
                flag_two, train_input_two = get_input_feature(file_path + file_two + '.wav')

                if flag_one == 1 or flag_two == 1:
                    train_input_one = train_input_two
                    train_label = 1

                train_input_one = np.reshape(train_input_one, (1, 300, 13))
                train_input_two = np.reshape(train_input_two, (1, 300, 13))
                train_inputs_one = np.row_stack((train_inputs_one, train_input_one))
                train_inputs_two = np.row_stack((train_inputs_two, train_input_two))
                train_labels = np.append(train_labels, train_label)
            except Exception as e:
                training_erro_time = d.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
                log.append(training_erro_time)
                log.append(e)
                continue
            if len(train_inputs_one) > batch_size:
                train_inputs_one = train_inputs_one[1:]
                train_inputs_two = train_inputs_two[1:]

                if mode == 'train':
                    train_inputs_one = train_inputs_one[0:int(0.7 * len(train_inputs_one))]
                    train_inputs_two = train_inputs_two[0:int(0.7 * len(train_inputs_two))]
                    train_labels = train_labels[0:int(0.7 * len(train_labels))]

                elif mode == 'validate':
                    train_inputs_one = train_inputs_one[int(0.7 * len(train_inputs_one)):]
                    train_inputs_two = train_inputs_two[int(0.7 * len(train_inputs_two)):]
                    train_labels = train_labels[int(0.7 * len(train_labels)):]

                yield [train_inputs_one, train_inputs_two], train_labels
                train_inputs_one = np.zeros((1, 300, 13))
                train_inputs_two = np.zeros((1, 300, 13))
                train_labels = np.array([])


training_start_time = d.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")

history = model.fit_generator(generate_batch('train'), steps_per_epoch=int(len(wav_label)/batch_size), epochs=epochs, verbose=1, validation_data=generate_batch('validate'), validation_steps=int(len(wav_label)/batch_size))

training_stop_time = d.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")

print("###############################################")
print("traning start time: ", training_start_time)
print("traning stop time: ", training_stop_time)
print("###############################################")

print(history.history.keys())
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# fig.savefig('./acc-lstm-lr0.0001-decay-6-batch60-epoch150.png.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
fig.savefig('./lstm-lr0.0001-decay-6-batch60-epoch150.png.png')

model.save('./lstm-lr0.0001-decay-6-batch60-epoch150.png.h5')

print(log)