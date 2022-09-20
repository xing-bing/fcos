import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import glob
import os

sets = ['train', 'val']

if __name__ == "__main__":
    base_dir = "dataset"
    for set in sets:
        with open(set + ".txt", 'w', encoding="utf-8") as f:
            path = base_dir + os.sep + set
            print(path)
            num_img = glob.glob(path + os.sep + "*.jpg")
            # 遍历所有文件
            for img in num_img:
                f.write(img)
                file_path = img.replace("jpg", "txt")
                readlines = open(file_path, 'r', encoding="utf-8")
                lines = [line.strip() for line in readlines.readlines()]
                for line in lines:
                    line = line.replace('(', "")
                    line = line.replace(')', "")
                    line = line.replace(' ', "")
                    f.write(" " + line)
                f.write("\n")
                readlines.close()
            f.close()
