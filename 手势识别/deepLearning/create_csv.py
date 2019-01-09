# !/usr/bin/env python

import sys
import os.path

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
#  philipp@mango:~/facerec/data/at$ tree
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#



fh = open("G:/ImageTest/face_test/orl_faces/at.txt",'w')

BASE_PATH = "G:/ImageTest/face_test/orl_faces"
SEPARATOR = ";"

label = 0
for dirname, dirnames, filenames in os.walk(BASE_PATH):
    for subdirname in dirnames:
        subject_path = os.path.join(dirname, subdirname)
        for filename in os.listdir(subject_path):
            abs_path = "%s/%s" % (subject_path, filename)
            ada_path=abs_path+SEPARATOR+subdirname[1:3]
            fh.write(ada_path)
            fh.write("\n")
            print("%s%s%s" % (abs_path, SEPARATOR, subdirname[1:3]))
        label = label + 1
fh.close()
