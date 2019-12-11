import os
import shutil

count = 372
dataDir = '/1TB/Datasets/Atari/data_acvp2/dataset/PongBowl2/'
# for folder in sorted(os.listdir(dataDir), reverse=True):
#     old_name = dataDir+folder
#     new_name = dataDir + str(count).zfill(5)
#     shutil.move(old_name, new_name)
#     count -= 1