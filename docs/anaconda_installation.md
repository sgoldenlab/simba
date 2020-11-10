# Step by step SimBA Installation in Anaconda environment

1. Download [Anaconda](https://www.anaconda.com/products/individual) and install the software into your machine.

2. Run Anaconda Navigator. The application looks like this:

![](/images/anacondanavigator.PNG)

3. Click on **Environments**, then click **Create** to create a new environment.

![](/images/anastep1.PNG)

4. Give it a name for your environment and choose Python 3.6. Then click **Create**

![](/images/anastep2.PNG)

5. Click on the environment that you just made. In this case, my new environment is *simba-test*. Then select `Open Terminal`

![](/images/anastep3.PNG)

6. In the terminal, type  `pip install simba-uw-no-tf`

![](/images/anastep4.PNG)

7. Once it is installed, uninstall shapely,

`pip uninstall shapely`

8. Then reinstall using the following command,

`conda install -c conda-forge shapely`

9. Type `simba` in the terminal to check if it works.
