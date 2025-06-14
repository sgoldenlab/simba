Global Configuration Options
====================================

SimBA has a few runtime configuration options which change the global behavior of SimBA.

These are managed by `python-dotenv` and are stored in the `simba/assets/.env <https://github.com/sgoldenlab/simba/blob/master/simba/assets/.env>`_ file of your python installation.

Sometimes, we may want to tweak these global settings - to unlock a few extra functionalities - or, to make sure that SimBA runs more reliably in specific hardware and operating system.

After `installing <https://simba-uw-tf-dev.readthedocs.io/en/latest/installation.html>`_, and before launching SimBA using `simba`, you can use the following commands:


LINUX
------------------------

.. code-block:: bash

    export PRINT_EMOJIS=False               #Turns of the use of emojis in the SimBA GUI
    export UNSUPERVISED_INTERFACE=True      #Enables GUI access to methods for unsupervised machine learning
    export NUMBA_PRECOMPILE=True            #Enable precompilation of Numba-based statistical methods. Results in slower SimBA load time but removed runtime cost associated with the first iteration run of any Numba decorated functions.
    export CUML=False                       #Enables GUI access to methods fitting supervised machine learning models using GPU device


Windows
------------------------

.. code-block:: bash

    set PRINT_EMOJIS=False                   #Turns of the use of emojis in the SimBA GUI
    set UNSUPERVISED_INTERFACE=True          #Enables GUI access to methods for unsupervised machine learning
    export NUMBA_PRECOMPILE=True             #Enable precompilation of Numba-based statistical methods. Results in slower SimBA load time but removed runtime cost associated with the first iteration run of any Numba decorated functions.
    set CUML=True                            #Enables GUI access to methods fitting supervised machine learning models using GPU device

Windows PowerShell
------------------------

.. code-block:: bash

   $env:PRINT_EMOJIS="False"                #Turns of the use of emojis in the SimBA GUI
   $env:UNSUPERVISED_INTERFACE="True"       #Enables GUI access to methods for unsupervised machine learning
   $env:NUMBA_PRECOMPILE="True"             #Enable precompilation of Numba-based statistical methods. Results in slower SimBA load time but removed runtime cost associated with the first iteration run of any Numba decorated functions.
   $env:CUML="True"                         #Enables GUI access to methods fitting supervised machine learning models using GPU device










