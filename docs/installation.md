# SimBA or SimBAxDLC?
**!!! IMPORTANT !!!**
You can choose to install SimBA as a standalone package or install SimBA with DeepLabCut integration.  

1) If you would like to be able to call DeepLabCut commands via the SimBA interface, and either have already installed DeepLabCut or would like to now install DeepLabCut on your local machine (requires a GPU), please install SimBAxDLC from the **master** branch.  Please see full instructions below.

2) If you do not want to use DeepLabCut on your local machine, and instead use Google Colab or have DeepLabCut installed elsewhere, please install SimBA from the **SimBA_no_DLC** branch. This does not require a GPU. Please see full instructions below.

# Requirements
1. [Python 3.6](https://www.python.org/downloads/release/python-360/)  **<-- VALIDATED WITH 3.6.0**
2. [Git](https://git-scm.com/downloads) 
3. [DeepLabCut](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md)  
4. [FFmpeg](https://m.wikihow.com/Install-FFmpeg-on-Windows)  **<-- NO JOKE, YOU MUST INSTALL THIS CORRRECTLY, CLICK LINK**

# Installing SimBA 

### Install SimBAxDLC with integrated DeepLabCut (use this installation method when running DeepLabCut locally using a GPU)  
Open bash or command prompt and run the following commands on current working directory

```
git clone -b master https://github.com/sgoldenlab/simba.git

pip install -r simba/simba/requirements.txt
```

### Install SimBA standalone package
Open bash or command prompt and run the following commands on current working directory

```
git clone -b SimBA_no_DLC https://github.com/sgoldenlab/simba.git

pip install -r simba/SimBA/requirements.txt
```

# How to launch SimBA

1. Open up command prompt in the SimBA folder

2. In the command prompt type
```
python SimBA.py
```
3. Hit `Enter`.

>*Note:* For this launch to work you need to [add python to the path](https://datatofish.com/add-python-to-windows-path/). 

# List of python dependencies
* [Pillow](https://github.com/python-pillow/Pillow)
* [deeplabcut](https://github.com/AlexEMG/DeepLabCut)
* [eli5](https://github.com/TeamHG-Memex/eli5)
* [imblearn](https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/imblearn)
* [imutils](https://github.com/jrosebr1/imutils)
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [numpy](https://github.com/numpy/numpy)
* [opencv_python](https://github.com/skvark/opencv-python)
* [openpyxl](https://github.com/chronossc/openpyxl)
* [pandas](https://github.com/pandas-dev/pandas)
* [scipy](https://github.com/scipy/scipy)
* [seaborn](https://github.com/mwaskom/seaborn)
* [sklearn](https://github.com/scikit-learn/scikit-learn)
* [tabulate](https://bitbucket.org/astanin/python-tabulate/src/master/)
* [tqdm](https://github.com/tqdm/tqdm)
* [xgboost](https://github.com/dmlc/xgboost)
* [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick)
