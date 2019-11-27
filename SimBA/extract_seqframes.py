# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:39:50 2019

@authors: Xiaoyu Tong, Jia Jie Choong, Simon Nilsson
"""
from PIL import Image
import io
import os
import numpy as np
import struct

class seqInfo:
    def __init__(self):
        self.version=0
        self.descr=''
        self.width=0
        self.height=0
        self.imageBitDepth=0
        self.imageBitDepthReal=0
        self.imageSizeBytes=0
        self.imageFormat=0
        self.numFrames=0
        self.trueImageSize=0
        self.fps=0.0
        self.codec=''

def extract_seqframescommand(filename):

    if filename:
        pathDir, filetype = filename.split('.')
        if not os.path.exists(pathDir):
            os.makedirs(pathDir)
        print(pathDir)

        fname = str(filename)
        extra = 4
        f = open(fname,'rb')
        info = readHeader(f)
        nframes = info.numFrames
        pos, frameSize = posFrame(f, nframes)
        fps = info.fps
        amount_of_frames = nframes
        print('The number of frames in this video = ',amount_of_frames)
        print('Extracting frames... (Might take awhile)')
        for i in range(nframes):
            f.seek(pos[i]+extra*(i+1), 0)
            imgdata = f.read(frameSize[i])
            img = Image.open(io.BytesIO(imgdata))
            img.save(pathDir + '\\' + os.path.basename(pathDir) + str(i) + '.png')
        print('All frames are extracted!')
    else:
        print('Please select a video to convert')

def posFrame(f,nframes):

    f.seek(1024,0)#header size is 1024. 
    pos=np.arange(nframes)
    frameSize=np.arange(nframes)
    pos[0]=1024
    extra=4#determined by manual test
    frameSize[0]=int.from_bytes(f.read(4),'little')
    for i in range(1,nframes):
        pos[i]=pos[i-1]+frameSize[i-1]+extra
        f.seek(frameSize[i-1]+extra,1)#find next frame
        frameSize[i]=int.from_bytes(f.read(4),'little')
    return pos,frameSize
        
def readHeader(f):

    info=seqInfo()
    f.seek(28,0)#read from the version info
    info.version=int.from_bytes(f.read(4),'little')
    f.read(4)#should be'1024'
    info.descr=f.read(512)#need dtype transform,not really supported for now
    info.width=int.from_bytes(f.read(4),'little')
    info.height=int.from_bytes(f.read(4),'little')
    info.imageBitDepth=int.from_bytes(f.read(4),'little')
    info.imageBitDepthReal=int.from_bytes(f.read(4),'little')
    info.imageSizeBytes=int.from_bytes(f.read(4),'little')
    info.imageFormat=int.from_bytes(f.read(4),'little')
    info.numFrames=int.from_bytes(f.read(4),'little')
    f.seek(4,1)#skip a 4-bytes blank
    info.trueImageSize=int.from_bytes(f.read(4),'little')
    fps=struct.unpack('<d',f.read(8))
    info.fps=fps[0]
    info.codec='imageFormat'+str(info.imageFormat)
    return info