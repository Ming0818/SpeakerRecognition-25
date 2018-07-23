# coding=UTF-8
##############
# Gain label #
##############

from OtUtils import PathConfig
import pandas as pd
import numpy as np

info_path = PathConfig.info_path
label_info_path = PathConfig.label_info_path

info = pd.read_csv(info_path)

processed_info = np.array([None, None, None])

for i in range(0, 36000, 90):
    for j in range(i, i+180):
        if j >= 36000:
            j = j - 36000
        print("j is: ", j)
        compare_info = []
        file_one = info.ix[i]['FILE_ID']
        user_one = info.ix[i]['USER_ID']
        file_two = info.ix[j]['FILE_ID']
        user_two = info.ix[j]['USER_ID']
        if user_one == user_two:
            label = 1
        else:
            label = 0

        if not processed_info.all():
            processed_info[0] = file_one
            processed_info[1] = file_two
            processed_info[2] = label
        else:
            compare_info.append(file_one)
            compare_info.append(file_two)
            compare_info.append(label)
            processed_info = np.row_stack((processed_info, compare_info))

processed_info = pd.DataFrame(processed_info, columns=["FILE_ID_ONE", "FILE_ID_TWO", "LABEL"])
processed_info.to_csv(label_info_path, index=False, sep=',')
