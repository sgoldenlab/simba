# Install SimBA using Anaconda Navigator

First, install Anaconda Navigator on your system. You can download Anaconda Navigator [HERE](https://www.anaconda.com/products/individual). For Microsoft Windows instructions, click [HERE](https://ultahost.com/knowledge-base/install-anaconda-on-windows/) or [HERE](https://www.geeksforgeeks.org/how-to-install-anaconda-on-windows/). For Linux/macOS, click [HERE](https://docs.anaconda.com/navigator/install/).

> [!TIP]
> If you need support and/or hit errors during the installation process, please each out to us by opening an [issue](https://github.com/sgoldenlab/simba/issues) and sending us a messeage on [Gitter](https://app.gitter.im/#/room/#SimBA-Resource_community).

> [!TIP]
> If you want to install SimBA using conda, but do not want to use the Anaconda Navigator, see [THESE](https://github.com/sgoldenlab/simba/blob/master/docs/installation_new.md) instructions.  


**STEP 1**. Run Anaconda Navigator. The application looks like this:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/anaconda_navigator_1.webp" />
</p>

**STEP 2**. On the left, click on the `Environments` tab, followed by the <kbd>Create</kbd> button highlighted in this image:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/anaconda_navigator_2.webp" />
</p>

**STEP 3**. Once clicked, its brings up teh below pop-up allowing you to enter the name of your new python conda environment, and which python version it should have. Here, we select python 3.6, and name the environment `simba_env`. Next, we click the <kbdCreate</kbd> button. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/anaconda_navigator_3.webp" />
</p>

**STEP 4**. Once complete, the new conda environment will be listed in the graphical interface, together with any other environments you have on your system:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/anaconda_navigator_4.webp" />
</p>

**STEP 5**. Each listed environment will have a little "play" button associated with it. Once we click on the play button, we will see some options. Go ahead and click on the `Open Terminal` option:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/anaconda_navigator_5.webp" />
</p>

**STEP 6**. This will bring up a terminal. In this terminal, you can see the name of your conda environment as the pre-fix of your path, highlighted with a red line on the left in the image below. Go ahead and typw `pip install simba-uw-tf-dev` (highlighted with a red line on the right in the image below) and hit <kbd>Enter</kbd>.
   After hitting <kbd>Enter</kbd>, SimBA will install on your system and you can follow the progress in the terminal window. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/anaconda_navigator_6.webp" />
</p>


**STEP 7**. Once installed, type `simba` in your ternimal window, and hit <kbd>Enter</kbd>, and SimBA will launch.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/anaconda_navigator_7.webp" />
</p>

>[!NOTE]
> SimBA may take a little time to launch depending in your computer, but you should eventually see [THIS](https://github.com/sgoldenlab/simba/blob/master/simba/assets/img/splash_2024.mp4) splash screen followed by [THIS](https://github.com/sgoldenlab/simba/blob/master/images/main_gui_frm.webp) main GUI window.



