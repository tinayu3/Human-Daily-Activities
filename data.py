import pandas as pd
import cv2
import numpy as np
import os

path1 = 'dataset/train.csv'
# Reading Data
df = pd.read_csv(path1)
# Extracting emotion data
df_y = df[['emotion']]
# Extracting pixels data
df_x = df[['pixels']]
# Write emotions to emotion.csv
df_y.to_csv('dataset/emotion.csv', index=False, header=False)
# Write pixels data to pixels.csv
df_x.to_csv('dataset/pixels.csv', index=False, header=False)

# Specify the path to store the image
path2 = 'face_images'
# Reading pixel data
data = np.loadtxt('dataset/pixels.csv')

# Fetch data by row
for i in range(data.shape[0]):
    face_array = data[i, :].reshape((48, 48)) # reshape
    cv2.imwrite(path2 + '//' + '{}.jpg'.format(i), face_array) 


def image_emotion_mapping(path):
    # Reading emotion files
    df_emotion = pd.read_csv('dataset/emotion.csv', header = None)
    # View all files in this folder
    files_dir = os.listdir(path)
    # Used to store picture names
    path_list = []
    # Used to store the emotion corresponding to the picture
    emotion_list = []
    # Traverse all files in this folder
    for file_dir in files_dir:
        # If a file is a picture, take out its file name and the corresponding emotion and put them into the two lists path_list and emotion_list respectively.
        if os.path.splitext(file_dir)[1] == ".jpg":
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])
            emotion_list.append(df_emotion.iat[index, 0])

    # Write the two lists into the image_emotion.csv file
    path_s = pd.Series(path_list)
    emotion_s = pd.Series(emotion_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['emotion'] = emotion_s
    df.to_csv(path+'\\image_emotion.csv', index=False, header=False)


def main():
    # Specify folder path
    train_set_path = 'face_images/train_set'
    verify_set_path = 'face_images/verify_set'
    image_emotion_mapping(train_set_path)
    image_emotion_mapping(verify_set_path)

if __name__ == "__main__":
    main()


