# Step by step SimBA Installation in Anaconda environment

1. Download [Anaconda](https://www.anaconda.com/products/individual) and install the software into your machine.

2. Run Anaconda Navigator. The application looks like this:

![](/images/anacondanavigator.PNG)

3. Click on **Environments**, then click **Create** to create a new environment.

![](/images/anastep1.PNG)

4. Give it a name for your environment and choose Python 3.6. Then click **Create**

![](/images/anastep2.PNG)

#### Note: If you do not have the option python 3.6. Just type `conda install python=3.6` in the terminal before step 6. 

5. Click on the environment that you just made. In this case, my new environment is *simba-test*. Then select `Open Terminal`

![](/images/anastep3.png)

6. In the terminal, type  `pip install simba-uw-tf-dev`

7. Type `simba` in the terminal to check if it works.


Author [Simon N](https://github.com/sronilsson), [JJ Choong](https://github.com/inoejj)
