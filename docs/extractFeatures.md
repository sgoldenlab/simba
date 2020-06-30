User can now use their own extract features script to run 

## Pre-requisite

1. The user should have an empty `__init__.py` file in the script path.

2. The name of the extract features script can be any name but it cannot contain any spaces. Eg: *arbitraryscriptname.py* is good, while *arbitrary script name .py* is bad.

3. In the python script, the main function should be name **extract_features_userdef** that allows the config ini file to pass in as an argument. Hence,
it should look something like this `def extract_features_userdef(inifile):`

## How to run your own script

1. Under **Extract Features**, check the checkbox `Apply user defined feature extraction script`.

2. Select your script by clicking on `Browse File`.

3. Click on `Extract Features` button to run your script through SimBA.

