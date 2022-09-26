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
from simba.drop_bp_cords import get_fn_ext

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
        pathDir, _, filetype = get_fn_ext(filename)
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
            img.save(os.path.join(pathDir, os.path.basename(pathDir) + str(i) + '.png'))
        print('All frames are extracted!')
    else:
        print('Please select a video to convert')

def posFrame(f,nframes):
    f.seek(1024, 0)  # header size is 1024.
    pos = np.arange(nframes, dtype=np.int64)
    frameSize = np.arange(nframes, dtype=np.int64)
    pos[0] = 1024
    extra = 4  # determined by manual test
    frameSize[0] = int.from_bytes(f.read(4), 'little')
    for i in range(1, nframes):
        pos[i] = pos[i - 1] + frameSize[i - 1] + extra
        f.seek(frameSize[i - 1] + extra, 1)  # find next frame
        frameSize[i] = int.from_bytes(f.read(4), 'little')
        while frameSize[i] == 0:
            frameSize[i] = int.from_bytes(f.read(4), 'little')
            pos[i] = pos[i] + extra
    return pos, frameSize
        
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


def convertseqVideo(videos,outtype='mp4',clahe=False,startF=None,endF=None):
    import os
    import io
    import cv2
    from tqdm import tqdm
    from PIL import Image
    from skimage import color
    from skimage.util import img_as_ubyte
    '''Convert videos to contrast adjusted videos of other formats'''
    ## get videos into a list in video folder
    videoDir = videos
    videolist = []
    for i in os.listdir(videoDir):
        if i.endswith('.seq'):
            videolist.append(os.path.join(videoDir,i))


    for video in videolist:
        vname = str(video)[:-4]
        f = open(video,'rb')
        info = readHeader(f)
        nframes = info.numFrames
        pos,frameSize = posFrame(f,nframes)
        fps=info.fps
        size = (info.width, info.height)
        extra=4
        if startF is None:
            startF=0
        if endF is None:
            endF=nframes
        if outtype=='avi':
            print('Coming soon')
        if outtype=='mp4':
            outname=os.path.join(vname + '.mp4')
            videowriter=cv2.VideoWriter(outname,cv2.VideoWriter_fourcc('m','p','4','v'),fps,size,isColor=True)
        for index in tqdm(range(startF,endF)):
            f.seek(pos[index]+extra*(index+1),0)
            imgdata=f.read(frameSize[index])
            image=Image.open(io.BytesIO(imgdata))
            image=img_as_ubyte(image)
            if clahe:
                image=cv2.createCLAHE(clipLimit=2,tileGridSize=(16,16)).apply(image)
            image=color.gray2rgb(image)
            frame=image
            videowriter.write(frame)
        f.close()
        videowriter.release()

    print('Finish conversion.')

