import numpy as np
import os
import cv2
from CAVnet_model import cavnet
import scipy.io as sio
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Tensorflow implementation of CAVnet')
parser.add_argument('--test_data_path', type=str, default=0,
                    help='Path of test input image')
parser.add_argument('--save_path', type=str, default=0,
                    help='The folder path to save output')
parser.add_argument('--save_mat', type=str, default=0,
                    help='The folder path to save output to mat')
parser.add_argument('--logdir', type=str, default=0,
                    help='Path of model weight')


if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.test_data_path
    save_path = args.save_path
    save_mat = args.save_mat
    log_dir = args.logdir

    img_row = None
    img_col = None
    img_channel = 1

    img_size = (img_row, img_col, img_channel)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_mat):
        os.mkdir(save_mat)
    model = cavnet(input_shape=img_size)
    model.load_weights(log_dir)

    file_list = os.listdir(data_path)
    for i in range(len(file_list)):
        print(i + 1)
        image = data_path + '\\'+file_list[i]
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        shape = img.shape
        if shape[0] % 16 != 0 or shape[1] % 16 != 0:
            new_size = (shape[0] // 16) * 16
            resize_img = cv2.resize(img, (new_size, new_size))
        else:
            resize_img = img
        name = str(file_list[i])
        out_name = name.split('.png')[0] + '_mask.png'
        Y1 = np.expand_dims(resize_img, 0)
        I1 = np.expand_dims(Y1, 3)
        meanI1 = np.mean(I1)
        stdI1 = np.std(I1)
        I1 = (I1 - meanI1) / stdI1
        pre = model.predict(I1, batch_size=1) * 255.

        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(np.uint8)
        pre = np.squeeze(pre)

        rgb = pre[..., [0, 3, 1, 2]].copy()
        OUTPUT_NAME = save_path + '\\'+out_name
        cv2.imwrite(OUTPUT_NAME, rgb[:, :, 1:4])
        mat_name = name.split('.png')[0] + '_mask.mat'
        images = rgb[:, :, :]
        imdict = {'masks': images}
        sio.savemat(save_mat + '\\'+mat_name, imdict)
