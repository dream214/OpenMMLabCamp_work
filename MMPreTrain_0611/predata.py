#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# Author：Sakura time:2023/6/8
import os
import random
import shutil

def splitData(imgPath,trainPath,testPath):
    for dirs in os.listdir(imgPath):
        for root, _, files in os.walk(os.path.join(imgPath,dirs)):
            allFiles = []
            for name in files:
                file=os.path.join(root,name)
                allFiles.append(file)
            print(allFiles)
            random.shuffle(allFiles)
            print(allFiles)
            for file in allFiles[:int(len(allFiles)*0.7)]:
                newFile=os.path.join(trainPath+'/'+dirs,os.path.basename(file))
                shutil.copy(file,newFile)
            for file in allFiles[int(len(allFiles)*0.7):]:
                newFile=os.path.join(testPath+'/'+dirs,os.path.basename(file))
                shutil.copy(file,newFile)
def renamedirs(trainPath,testPath):
    with open('数字和水果中文名称对应.txt','w') as f:
        for index,dirs in enumerate(os.listdir(trainPath)):
            os.rename(os.path.join(trainPath,dirs),os.path.join(trainPath,str(index)))
            f.write(dirs+' '+str(index)+'\n')
            os.rename(os.path.join(testPath, dirs), os.path.join(testPath, str(index)))


if __name__=="__main__":
    splitData('数据集',
              'data/train',
              'data/test')
    renamedirs('data/train',
               'data/test')