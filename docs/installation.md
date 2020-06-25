# SimBAxTF or SimBA?
**!!! IMPORTANT !!!**
You can choose to install SimBA as a standalone package or install SimBA with [TensorFlow](https://github.com/tensorflow/tensorflow) integration.  

1) If you would like to be able to call [DeepLabCut](https://github.com/AlexEMG/DeepLabCut) or [DeepPoseKit](https://github.com/jgraving/DeepPoseKit) commands via the SimBA interface (whuch requires a local GPU), please install SimBAxTF from the **master** branch. Please see the SimBA tutorials on [DeepLabCut](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md) or [DeepPoseKit](https://github.com/sgoldenlab/simba/blob/master/docs/DeepPoseKit_in_SimBA.md) for information on what it means to run DeepLabCut and DeepPoseKit within the SimBA GUI. **See full installation instructions below.**

2) If you do **not** want to use TensorFlow through SimBA on your local machine, and instead have DeepLabCut/DeepPoseKit installed elsewhere, please install SimBA from the **SimBA_no_TF** branch. This does not require a GPU, or local installations of DeepLabCut or DeepPoseKit. Please see full instructions below. This version of SimBA includes **all functionalities** of SimBAxTF, except for the ability to generate pose-estimation models through the SimBA GUI. **Pose-estmation model results can still be imported and analysed**.

# Requirements
1. [Python 3.6](https://www.python.org/downloads/release/python-360/)  **<-- VALIDATED WITH 3.6.0**
2. [Git](https://git-scm.com/downloads) 
3. [FFmpeg](https://m.wikihow.com/Install-FFmpeg-on-Windows)

# Installing SimBA Option 1 (RECOMMENDED!)

### Install SimBAxTF with integrated TensorFlow (use this installation method when running DeepLabCut, DeepPoseKit, or SLEAP locally using a GPU)
Open bash or command prompt and run the following commands on current working directory

```
pip install simba-uw-tf
```

### Install SimBA standalone package (without TensorFlow or integrated DeepLabCut/DeepPoseKit support)
Open bash or command prompt and run the following commands on current working directory

```
pip install simba-uw-no-tf
```

# How to launch SimBA ( installed using pip install simba-uw-tf)

1. Open up command prompt anywhere.

2. In the command prompt type
```
simba
```
3. Hit `Enter`.

>*Note:* If you installed SimBA on a virtual environment (anaconda), after installation, you may have to run run `conda install shapely` for SimBA to work.


# Installing SimBA Option 2

### Install SimBAxTF with integrated TensorFlow (use this installation method when running DeepLabCut or DeepPoseKit locally using a GPU)
Open bash or command prompt and run the following commands on current working directory

```
git clone -b master https://github.com/sgoldenlab/simba.git

pip3 install -r simba/simba/requirements.txt
```

### Install SimBA standalone package (without TensorFlow or integrated DeepLabCut/DeepPoseKit support)
Open bash or command prompt and run the following commands on current working directory

```
git clone -b SimBA_no_TF https://github.com/sgoldenlab/simba.git

pip3 install -r simba/SimBA/requirements.txt
```

# How to launch SimBA (installing by cloning)

1. Open up command prompt in the SimBA folder

2. In the command prompt type
```
python SimBA.py
```
3. Hit `Enter`.

>*Note:* For this launch to work you need to [add python to the environmental path](https://datatofish.com/add-python-to-windows-path/). 

# python dependencies

| package  | ver. |
| ------------- | ------------- |
| [Pillow](https://github.com/python-pillow/Pillow) | 5.4.1  |
| [deeplabcut](https://github.com/AlexEMG/DeepLabCut) | 2.0.9 |
| [eli5](https://github.com/TeamHG-Memex/eli5)  | 0.10.1 |
| [imblearn](https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/imblearn) | 0.5.0 |
| [imutils](https://github.com/jrosebr1/imutils)  | 0.5.2  |
| [matplotlib](https://github.com/matplotlib/matplotlib)  | 3.0.3  |
| [Shapely](https://shapely.readthedocs.io/en/latest/index.html)  | 1.6.4.post2 |
| [deepposekit](https://github.com/jgraving/DeepPoseKit) | 0.3.5 |
| [dtreeviz](https://github.com/parrt/dtreeviz)   | 0.8.1  |
| [opencv_python](https://github.com/skvark/opencv-python)| 3.4.5.20 |
| [numpy](https://github.com/numpy/numpy)|1.18.1 |
| [imgaug](https://imgaug.readthedocs.io/en/latest/)| 0.4.0 |
| [pandas](https://github.com/pandas-dev/pandas)| 0.25.3 |
| [scikit_image](https://scikit-image.org/)| 0.14.2  |
| [scipy](https://github.com/scipy/scipy)| 1.1.0  |
| [seaborn](https://github.com/mwaskom/seaborn)| 0.9.0  |
| [sklearn](https://github.com/scikit-learn/scikit-learn)| 1.1.0  |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn)| 0.22.1 |
| [tensorflow_gpu](https://github.com/tensorflow/tensorflow)| 0.14.1 |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn)| 0.22.1 |
| [tqdm](https://github.com/tqdm/tqdm)| 4.30.0 |
| [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick)| 0.9.1 |
| [xgboost](https://github.com/dmlc/xgboost)| 0.9 |
| [tabulate](https://bitbucket.org/astanin/python-tabulate/src/master/)| 0.8.3 |
| [tables](https://www.pytables.org/)| ≥ 3.5.1 |


