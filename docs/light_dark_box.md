# <p align="center"> Light dark box analysis in SimBA </p>


## GENERAL REQUREMENTS 

>[!NOTE]
>This analysis does **NOT** require the creation of a SimBA project. For data, all that is needed is a directory containing CSV files presenting pose-estimation data from DeepLabCut. The CSV files are expected to be in [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/light_dark_box_expected_input.csv) format.
>However, we do need to install SimBA as docuemnted [HERE](https://github.com/sgoldenlab/simba/edit/master/docs/installation_new.md) or [HERE](https://simba-uw-tf-dev.readthedocs.io/en/latest/pip_installation.html).

### STEP 1 - GET THE REQUIRED PYTHON FILE

1. Locate the `light_dark_box_analyzer.py` script. It is located [HERE](https://github.com/sgoldenlab/simba/blob/master/simba/data_processors/light_dark_box_analyzer.py). It could also be found at path `simba.data_processors.light_dark_box_analyzer.LightDarkBoxAnalyzer` in the github repository. If you already have a SimBA environment installed on your computer, you can locate the file by (i) typing `pip show simba-uw-tf-dev` in your SimBA environment, (ii) enter the path that is shown (as in the scrrengrab below) into a file explorer, and navigate to the `simba/data_processors/light_dark_box_analyzer` inside that shown directory.

![image](https://github.com/user-attachments/assets/46304438-4eb6-4de0-8f5e-4c57b60e2a99)

2. Copy this file (or paste the content of this file into an empty file named `light_dark_box_analyzer.py`) to an accessable location on your computer. For example, I have copied this file to location `C:\projects\light_dark_box\light_dark_box_analyzer.py` for writing this documentation.  

### STEP 2 - FORMULATE A COMMAND

To run the file above, we need to formulate a command line text string specifying how we want the command to run. This inlcudes things like where our deeplabcut data is saved, where we want to save the output, the frame rate of out videos, and a few other things detaile below:

`data_dir`: The path to the directory containing CSV files presenting pose-estimation data from DeepLabCut. In my example this is `C:\projects\light_dark_box\data`

`save_path`: The path to the directory containing CSV files presenting pose-estimation data from DeepLabCut.



python light_dark_box_analyzer.py --data_dir 'D:\light_dark_box\project_folder\csv\input_csv' --save_path "D:\light_dark_box\project_folder\csv\results\light_dark_data.csv" --body_part nose --fps 29 --threshold 0.01






