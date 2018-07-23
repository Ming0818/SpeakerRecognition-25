# coding=UTF-8
#####################################
# preprocess：denoise all wav-files #
#####################################

from OtUtils import enhance_speach, PathConfig
import os

input_file_path = PathConfig.input_file_path
output_file_path = PathConfig.output_file_path
file_name = os.listdir(input_file_path)

step = 0

for file in file_name:
    input_file_name = input_file_path+file
    output_file_name = output_file_path+file
    step += 1
    print(step)
    # print(input_file_name)
    # print(output_file_name)
    enhance_speach.process(input_file_name, output_file_name)