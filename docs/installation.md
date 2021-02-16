# SimBA Installation

# Requirements
1. [Python 3.6](https://www.python.org/downloads/release/python-360/)  **<-- VALIDATED WITH 3.6.0**
2. [Git](https://git-scm.com/downloads) 
3. [FFmpeg](https://m.wikihow.com/Install-FFmpeg-on-Windows)
4. Microsoft Windows operating system

# Installing SimBA Option 1 

### Install SimBAxTF 
Open bash or command prompt and run the following commands on current working directory

```
pip install simba-uw-tf
```

### Install SimBAxTF-development version
Open bash or command prompt and run the following commands on current working directory

```
pip install simba-uw-tf-dev
```

>Note: If you are seeing error messages related to some dependency conflicts, then you need to either downgrade your pypi package or instruct SimBA to ignore these dependency conflicts - either works. To find more information on how to do this, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/FAQ.md#when-i-install-or-update-simba-i-see-a-bunch-or-messages-in-the-console-telling-there-has-been-some-dependency-conflicts-the-messages-may-look-a-little-like-this)


# How to launch SimBA ( installed using pip install simba-uw-tf)

1. Open up command prompt anywhere.

2. In the command prompt type
```
simba
```
3. Hit `Enter`.

>*Note:* If you installed SimBA on a virtual environment (anaconda), after installation, you may have to run run `conda install shapely` for SimBA to work.


# Installing SimBA using Anaconda
Click [here](/docs/anaconda_installation.md) for a detail step by step guide on how to install using anaconda.

1. Open up terminal of your environment

2. In the terminal type 

`pip install simba-uw-tf`

3. It will error out when running simba. To fix it, first uninstall shapely.

`pip uninstall shapely`

4. Then, install shapely with conda command:

`conda install -c conda-forge shapely`


# Installing on MacOS

This is not recommended but it is possible.

### Requirements
- XCode installed
- Homebrew
- ffmpeg
- Python 3.6
- Anaconda

## Installation process

1. Create an environment for simba using anaconda terminal.

2. In the terminal type,
`pip install simba-uw-no-tf`

3. Then,
`conda install -c anaconda python.app`

4. Then,
`conda install matplotlib`

5. Then,
`conda uninstall shapely`

6. Then,
`conda install -c conda-forge shapely`

7. Then,
`pip install shap`

8. Lastly,
`pip install h5py`

9. In the terminal, type in `simba` to test if it works.

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
| [dash](https://github.com/plotly/dash/)| 1.14.0 |
| [dash color picker](https://github.com/vivekvs1/dash-color-picker/)| 0.0.1 |
| [dash daqs](https://dash.plotly.com/dash-daq)| 0.5.0 |
| [h5py](https://github.com/h5py/h5py)| 2.9.0 |
| [numba](https://github.com/numba/numba)| 0.48.0 |
| [numexpr](https://github.com/pydata/numexpr)| 2.6.9 |
| [plotly](https://github.com/plotly)| 4.9.0 |
| [statsmodels](https://github.com/statsmodels/statsmodels)| 0.9.0 |
| [cefpython3](https://github.com/cztomczak/cefpython)| 66.0 |
| [pyarrow](https://github.com/apache/arrow/tree/master/python/pyarrow)| 0.17.1 |
| [shap](https://github.com/slundberg/shap)| 0.35.0 |



