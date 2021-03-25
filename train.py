from os import path, listdir, mkdir, remove
from shutil import rmtree
import cv2 
import subprocess


TRAIN_DIR = 'train'
TRAIN_GRAY_DIR = 'train_gray'
OPENCV_BIN_DIR = 'e:\\Work\\Libs\\opencv-3.4.13\\opencv\\build\\x64\\vc15\\bin\\'
BG_DIRS = ['actual_bg']

MAIN_DIR = TRAIN_DIR

def convert_to_gray():
    rmtree(TRAIN_GRAY_DIR, ignore_errors=True)
    mkdir(TRAIN_GRAY_DIR)
    
    for i in range(0, 7):
        name = 'bg' if i == 0 else str(i)

        dir = path.join(TRAIN_DIR, name)
        graydir = path.join(TRAIN_GRAY_DIR, name)
        mkdir(graydir)

        for file in listdir(dir):
            filepath = path.join(dir, file)
            graypath = path.join(graydir, file)

            img = cv2.imread(filepath)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            cv2.imwrite(graypath, img_gray)

def generate_info():
    info_file = f'all.info'
    with open(info_file, 'w') as f:
        for i in range(1, 7):
            dir = path.join(MAIN_DIR, str(i))

            for file in listdir(dir):
                filepath = path.join(dir, file)

                img = cv2.imread(filepath)
                h, w, _ = img.shape
                f.write(f'{filepath} 1 0 0 {w} {h}\n')

    bg_file = f'all_bg.txt'
    with open(bg_file, 'w') as f:
        for bg_dir in BG_DIRS:
            dir = path.join(MAIN_DIR, bg_dir)
            for file in listdir(dir):
                filepath = path.join(dir, file)
                f.write(f'{filepath}\n')


def create_samples():
    createsamples_app = path.join(OPENCV_BIN_DIR, 'opencv_createsamples')
    traincascade_app = path.join(OPENCV_BIN_DIR, 'opencv_traincascade')

    rmtree('data', ignore_errors=True)
    mkdir('data')

    i = 'all'
    info_file = f'{i}.info'
    vec_file = f'{i}.vec'
    bg_file = f'{i}_bg.txt'
    data_dir = f'data\\{i}'

    mkdir(data_dir)

    with open(info_file, 'r') as f:
        num = len(f.readlines())

    with open(bg_file, 'r') as f:
        num_neg = len(f.readlines())

    subprocess.run([createsamples_app, '-info', info_file, '-num', str(num), '-vec', vec_file])
    subprocess.run([traincascade_app, '-data', data_dir, '-vec', vec_file, '-bg', bg_file, '-numPos', str(int(num*0.95)), '-numNeg', str(num_neg), '-numStages', '10', '-minHitRate', '0.999', '-maxFalseAlarmRate', '0.1', '-featureType', 'LBP'])

def remove_file(name: str):
    tmp_files = ['.info', '.vec', '_bg.txt']
    for ext in tmp_files:
        file = path.join(name, ext)
        if path.exists(file):
            remove(file)

def cleanup():
    for i in range(1, 7):
        remove_file(str(i))
    remove_file('all')
    rmtree(TRAIN_GRAY_DIR, ignore_errors=True)

cleanup()
convert_to_gray()
generate_info()
create_samples()
cleanup()