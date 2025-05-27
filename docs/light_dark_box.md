# <p align="center"> Light dark box analysis in SimBA </p>


## GENERAL REQUREMENTS 

>[!NOTE]
>This analysis does **NOT** require the creation of a SimBA project. For data, all that is needed is a directory containing CSV files presenting pose-estimation data from DeepLabCut. The CSV files are expected to be in [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/light_dark_box_expected_input.csv) format.
>
>However, we do need to install SimBA as documented [HERE](https://github.com/sgoldenlab/simba/edit/master/docs/installation_new.md) or [HERE](https://simba-uw-tf-dev.readthedocs.io/en/latest/pip_installation.html).
>
>Make sure you have SimBA version 3.1.7 or later. 

### STEP 1 - GET THE REQUIRED PYTHON FILE

**1.** Locate the `light_dark_box_analyzer.py` script. It is located [HERE](https://github.com/sgoldenlab/simba/blob/master/simba/data_processors/light_dark_box_analyzer.py). It could also be found at path `simba.data_processors.light_dark_box_analyzer.LightDarkBoxAnalyzer` in the github repository. If you already have a SimBA environment installed on your computer, you can locate the file by (i) typing `pip show simba-uw-tf-dev` in your SimBA environment, (ii) enter the path that is shown (as in the scrrengrab below) into a file explorer, and navigate to the `simba/data_processors/light_dark_box_analyzer` inside that shown directory.

![image](https://github.com/user-attachments/assets/46304438-4eb6-4de0-8f5e-4c57b60e2a99)

**2.** Copy this file (or paste the content of this file into an empty file named `light_dark_box_analyzer.py`) to an accessable location on your computer. For example, I have copied this file to location `C:\projects\light_dark_box\light_dark_box_analyzer.py` for writing this documentation.  

### STEP 2 - FORMULATE A COMMAND

To run the file above, we need to formulate a command line text string specifying how we want the python .py file to run. This inlcudes things like where our deeplabcut data is saved, where we want to save the output, the frame rate of out videos, and a few other things detailed below:

**data_dir**: The path to the directory containing CSV files presenting pose-estimation data from DeepLabCut. In my example below I will use `"C:\projects\light_dark_box\data"` (note the use of qoutation marks around the path).

**save_path**: The location where we wan to store th CSV data results. In my example below I will use  `"C:\projects\light_dark_box\light_dark_data.csv"` (note the use of qoutation marks around the path).

**body_part**: The name of the body-part for which we want to use to proxy the animals absent or presence from the light- and dark-box. In my example below I will use `center`.

**fps**: The frame rate at which the the videos (which was analysed for pose-estimation) where recorded. In my example below I will use `29.97` (note that this calue can also be passed as an integer (e.g., `30`, `10` `10.12`, `30.67` will all work).

**threshold**: If the pose-estimation probability value for the specified body-part falls below this value, animal assumed to be in the dark box. If above or at this value, animal is assumed to be in the light box. The value should be between 0 and 1. In my example below I will use `0.01`.

**minimum_episode_duration**: Sometimes we may get a spurruious frame or two when the probability of the specified bosy-part is high or low, leading to odd observations of the animal visiting the ligh- or dark box for a few milliseconds. We want to remove these spurruious observations. This value represents the shortest allowed duration that the animal can visit a compartment in seconds. n my example below I will use `1` for one second. 

Using the above settings, my command line command will read:

```bat
python light_dark_box_analyzer.py --data_dir "C:\projects\light_dark_box\data" --save_path "C:\projects\light_dark_box\light_dark_data.csv" --body_part center --fps 29.97 --threshold 0.01 --minimum_episode_duration 1
```

I often put this text string together in some notepad or other texteditor document, and later copy-paste it to the termina. 

### STEP 3 - RUN THE COMMAND.

**1.** Open a command line terminal and navigate to the directory which stores `light_dark_box_analyzer.py`

**2.** Activate your SimBA environment. If you have a conda environment called `simba` with simba installe dinside it, run:

```bat
conda activate simba
```

**3.** Copy paste the command from the end of **STEP 2** inside the teminal window. You can follow the progress in the terminal window. 

In the below video, you can see my completing these three things to analyze 15 DeppLabCut pose-estimation files:


https://github.com/user-attachments/assets/583f4d75-09dc-4249-9ea3-80a871faf74d

The results are saved in a single CSV file at location "C:\projects\light_dark_box\light_dark_data.csv" indexed according to video name where each row is a episode in the ligh or dark compartment together with information of episode start time, end time, duration, start frame and end frame etc. For a small expected output example, see [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/light_dark_data.csv) file.






















