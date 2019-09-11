# Tutorial

### Pipeline breakdown:

For processing a tracking dataset, the pipelines is split into ? sections. These sections are listed below along with the functions that belong to each section:


#### Part 1: Create a new project
- In here you set up your project name, directories, and videos settings
- Then, generate project config
- Import videos into project folder
- Import DLC Tracking Data (if have any)
- Extract Frames into project folder

#### Part 2: Load project.ini file
- First you load the project.ini that was generated when you created your new project
- set video parameters and save it as csv
- Run outlier correction
- Extract Features
- Label Behavior
- Train Machine Model
- Run Machine Model
- Analyze Machine Results
- Plot Skleran Results
- Plot Graphs
- Merge Frames
- Create Video

## Part 1: Create a new project
This sections create a new project for your tracking analysis.

#### Step 1: Generate Project Config
1. Go to `File` and click on `Create a new project` The following windows will pop up.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/createproject.PNG" width="500" height="600" />

2. `Project Path` is the main working directory that your project will be created. Click on `Browse Folder` and select your working directory

3. `Project Name` is the name for your project. *Keep in mind that the project name cannot contain any spaces, use underscore "_" instead* 

4. Under `SML Settings`, put in the number of predictive classifiers that you wished. For an example, if you had three behaviors in your video, put 3 in the entry box.

5. Click <img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/addclassifier.PNG" width="150" height="50" /> and it creates a row as shown in the following image.
If you are generating a brand new model, just fill in the entry box and leave the path as it is (*No classifier is selected*). Repeat this step for every behaviors.
> **Note**: Make sure to click the button once and finished it before clicking it again. Do not click to add multiple rows before filling up the information.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/classifier1.PNG" width="700" height="130" />

6. For existing model, you can import the models by clicking `Browse` and select your *.sav* file. At the end, it should look like the following image.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/classifier2.PNG" width="700" height="160" />

