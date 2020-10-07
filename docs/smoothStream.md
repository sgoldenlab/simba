# [SmoothStream](https://github.com/CT83/SmoothStream)
Webcam and PiCamera Streaming over the Network with Python

## Installing
1. Go to CT83's [SmoothStream github](https://github.com/CT83/SmoothStream), and download the zip file and follow the instructions on his github.

2. Extract the SmoothStream-master folder.

3. Do this on the streamer's machine and the viewer's machine.

## Tutorial

### Streamer

1. Get the ip address of the viewer's machine. The easiest way of doing this is go to command prompt and type `ipconfig`, and get the IPv4 address.

2. On the streamer's machine, open command prompt in SmoothStream directory and type 
```
python Streamer.py -s 172.18.161.222
```

In this case, the ip address of my viewer's machine is 172.18.161.222.

## Viewer

1. On the viewer's machine, open command prompt in the  SmoothStream directory and type 
```
python StreamViewer.py
```
