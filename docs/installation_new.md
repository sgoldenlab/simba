# SimBA Installation

## Requirements
* [Python 3.6](https://www.python.org/downloads/release/python-360/)
>[!NOTE]  
> SimBA is validated using python 3.6, and the developers maintain SimBA mainly using 3.6. However, you can use python 3.10 if necessery. If you encounter bugs using 3.10, please each out to us by opening an [issue](https://github.com/sgoldenlab/simba/issues) and sending us a messeage on [Gitter](https://app.gitter.im/#/room/#SimBA-Resource_community).
* [FFmpeg](https://www.ffmpeg.org/)
See installation instructions for [Windows](https://m.wikihow.com/Install-FFmpeg-on-Windows), [MacOS/Linux](https://www.ffmpeg.org/download.html)

## Install options:

* In main Python environment:
* Using conda:
* Using Anaconda Navigator:
* Uing Python venv:


### Option 1: Install SimBA using Anaconda (recommended)

Click [here](/docs/anaconda_installation.md) for a detail step by step guide on how to install using anaconda.

1. Once conda is installed, create a new python3.6 environment

`````````
conda create -n my_simba_env python=3.6 anaconda -y
`````````

.... or python3.10, if really needed:


`````````
conda create -n my_simba_env python=3.10 anaconda -y
`````````


2. Enter the conda environment by typing:

`````````
conda activate my_simba_env
`````````


3. Install SimBa in the new `my_simba_env` environment by typing:

```
pip install simba-uw-tf-dev
```

or, if in **python 3.10** and youre hitting errors:
````
pip install simba-uw-tf-dev --ignore-installed
````

4) Now launch SImBA by opening a command prompt and type:

`````````
simba
`````````

.. and hit the  <kbd>Enter</kbd> key. Note: SimBA may take a little time to launch depending in your computer, but you should eventually see a splash screen and the main GUI windows below

> [!TIP]
> You can also use the Anaconda Navigoator GUI to get this done, which creates conda environments through a graphical interface rather than through the command line. You can read about how to install SimBA using the Anaconda Navigator [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/anaconda_installation.md).


### Option 2: Install SimBA in main python.

1). Open bash or command prompt and type the following commands on current working directory

```
pip install simba-uw-tf-dev
```

>Note: If you are seeing error messages related to some dependency conflicts, then you need to either downgrade your pypi package or instruct SimBA to ignore these dependency conflicts - either works. To find more information on how to do this, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/FAQ.md#when-i-install-or-update-simba-i-see-a-bunch-or-messages-in-the-console-telling-there-has-been-some-dependency-conflicts-the-messages-may-look-a-little-like-this).

2) Now launch SImBA by opening a command prompt and type:

`````````
simba
`````````

.. and hit the  <kbd>Enter</kbd> key. Note: SimBA may take a little time to launch dependning in your computer CPU, but you should eventually see a splash screen and the main GUI windows below


### Option 3: Install SimBA using python venv

I honestly haven't used this method much. But these are the steps that I have had reported to me that users had to go through: 

1. Open bash or command prompt and run the following commands on current working directory 
``` 
python -m venv venv
```
or (to make sure your virtuall environment is python 3.6 if you have multiple python versions in your machine)
```
py -3.6 -m venv venv
```

2. Then activate virtual environment by 
```
venv\Scripts\activate
```
3. Make sure you are using the latest pip version and setup tools
```
python -m pip install --upgrade pip
pip uninstall setuptools
pip install setuptools
```
4. Install simba 
```
pip install simba-uw-tf-dev
```

5. Fix some package version
```
pip3 uninstall pyparsing
pip3 install pyparsing==2.4.7
```

6. Now you can launch simba in the terminal with. 
```
simba
```
