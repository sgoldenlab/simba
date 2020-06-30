User can now use their own extract features script to run. Click [here](https://osf.io/cmkub/) to download the sample script.

## Pre-requisite

![](/images/effolder.PNG)

1. The user should have an empty `__init__.py` file in the script path.

2. The name of the extract features script can be any name but it cannot contain any spaces. Eg: *arbitraryscriptname.py* is good, while *arbitrary script name .py* is bad.

3. The name of the folder that contains the script should not contain any spaces too.

4. In the python script, the main function should be name **extract_features_userdef** that allows the config ini file to pass in as an argument. Hence,
it should look something like this `def extract_features_userdef(inifile):`

![](/images/defextractf.PNG)

## How to run your own script

![](/images/extractfusrdef.PNG)

1. First, load your project and navigate to the `[Extract Features]` tab.

2. Under **Extract Features**, check the checkbox `Apply user defined feature extraction script`.

3. Select your script by clicking on `Browse File`.

4. Click on `Extract Features` button to run your script through SimBA.

