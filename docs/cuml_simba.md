


####
**1)** In Linux, be in a python **3.10** environment with SimBA installed. you can create a SimBA python 3.0 environment as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/installation_new.md).

>[!NOTE]  
> Microsoft Windows WSL environment should work.
>
> To create the conda environment use e.g., `conda create -n simba_3_10 python=3.10 anaconda -y`.
>
> Use SImBA version at or above ``2.9.4``.

**2)** Install the cuml-cu12 (which is not present in the standard simba requirements.txt. After activating the SimBA python 3.10 environment created in step 1 type:

```
pip install cuml-cu12==24.12.0
```

**3)** Next, we need to tell SimBA to look for cuml-cu12 (which was installed in the prior step) when booting up. To do this, in the command line, type:

```
export CUML=True
```
![image](https://github.com/user-attachments/assets/c6380386-6c45-480e-9c8e-ef11f6b1297b)

and hit <kbd>ENTER</kbd>.

**4)** Next, launch SimBA with `simba`. If everything has gone to plan you should see the below beeing printed out. Specifically, you should see ``SimBA CUML enabled.`` followed by ``'CUML': True``, as in the below screengrab. 

![image](https://github.com/user-attachments/assets/66d13d3d-b02a-4f3b-adfd-4016747cbf5e)

Set you global machine learning paramaters as usual in SimBA as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#train-single-model), or create multiple model config files as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#to-train-multiple-models) to train multiple models. However, don't click to train the model(s) yet, we need to modify one thing in the files to tell them to use the GPU libraries that we just have imported into SimBA.  

**5)**

##### TRAINING A SINGLE MODEL
If you are training from the global environment, open the `project_config.ini` file and add one parameter to the [create ensemble settings] section. Add:

  ``cuda = True``

  as in screengrab below":

  ![image](https://github.com/user-attachments/assets/7c2e7e8d-9056-4ec4-bd55-fdb31328c3e3)

and save the file. 


##### TRAINING MULTPLE MODELS
If you are training multiple models, open each of the CSV files in the project_folder/configs and add one header named ``cuda`` and set the value to TRUE, as in the screengrab below:

![image](https://github.com/user-attachments/assets/d2302531-876c-44e6-8e17-d1919e75d74d)

> [!NOTE]  
> Microsoft Windows WSL environment should work.
>




