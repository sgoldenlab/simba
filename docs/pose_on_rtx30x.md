### Pose estimation on newer GPUs

With the newer GPUs (e.g., NVIDEA Rtx3080 ) you can no longer use Tensorflow (TF) version 1.x, which is what earlier versions of DeepLabCut depend on. Instead, you have to use TF version 2.x, and this requires a different set of package versions than previously. Here is our protocol for getting TF and DLC to work on our **Microsoft Windows** NVIDEA Rtx3080 supported machines. 

##### 1.	Download and install CUDA version 11.0 **and** CUDA version 11.1 for Windows
##### 2.	Download cudnn version 8.0.4.30 for windows
##### 3.	Install cudnn in the CUDA 11.0 version 
>Note: The CUDA default installations directories on Windows is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0` and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1`, respectively. 

##### 4.	Add the CUDA bin folder to the windows environment paths. 
>Note: The Windows environment paths may look something like this: <p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Rtx3090_1.png" />
</p>

##### 5.	Create a new conda environment – python 3.6 should work. In your conda environment, install the following python packages:

* tensorflow-gpu==2.4

* keras==2.3.1

* scipy==1.2

* protobuf==3.6.0

* tf_slim==1.1.0 

* tf-nightly==2.5.0.dev20210128

Alternatively copy past this in the terminal: 

`pip install tensorflow-gpu==2.4 keras==2.3.1 scipy==1.2 protobuf==3.6.0 tf_slim==1.1.0 tf-nightly== 2.5.0.dev20210128`


##### 6.	In your conda environment, install `deeplabcutcore` with the below command. For more information, see the [DeepLabCut documentation](https://github.com/DeepLabCut/DeepLabCut-core/blob/tf2.2alpha/Colab_TrainNetwork_VideoAnalysis_TF2.ipynb):

pip install git+https://github.com/DeepLabCut/DeepLabCut-core.git@tf2.2alpha

##### 7.	In your **CUDA 11.1** folder, locate ptxas.exe. Copy this file and use it to replace ptxas.exe in your **CUDA 11.0** version. 


##### 8.	At the time of writing, I don’t think `deeplabcutcore` has a graphical interface, so we have to run it by the command line. We run the DLC model with the script like this. It can be necessary to specify the `per_process_gpu_memory_fraction` argument as less than 1, as the default settings may consume all memory and kill the program. 

````
import deeplabcutcore as deeplabcut
import tensorflow as tf
import sys

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def perform_pose(project_path, videos_path):
    deeplabcut.analyze_videos(project_path, videos_path, videotype='mp4', shuffle=1, save_as_csv=True)

if __name__ == "__main__":
    project_path = DLC PROJECT CONFIG YAML PATH E.G r"C:\Users\Windows\Desktop\Project1-SN-2020-04-07\config.yaml"
    videos_path = LIST CONTAINING STRING PATH TO THE FOLDER WHERE THE VIDEOS TO ANALYZE ARE LOCATED  E.G [“C:\Users\Windows\Desktop\MyVideos”]
    perform_pose(project_path, videos_path)
    print('Video analysis complete')

`````
