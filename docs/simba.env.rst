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
    export MP_START_METHOD=spawn            #Start method for multiprocessing ('fork', 'spawn' or 'forkserver'). Set to 'spawn' if visualizations crash with an [xcb] error (see below).


Windows
------------------------

.. code-block:: bash

    set PRINT_EMOJIS=False                   #Turns of the use of emojis in the SimBA GUI
    set UNSUPERVISED_INTERFACE=True          #Enables GUI access to methods for unsupervised machine learning
    set NUMBA_PRECOMPILE=True                #Enable precompilation of Numba-based statistical methods. Results in slower SimBA load time but removed runtime cost associated with the first iteration run of any Numba decorated functions.
    set CUML=True                            #Enables GUI access to methods fitting supervised machine learning models using GPU device

Windows PowerShell
------------------------

.. code-block:: bash

   $env:PRINT_EMOJIS="False"                #Turns of the use of emojis in the SimBA GUI
   $env:UNSUPERVISED_INTERFACE="True"       #Enables GUI access to methods for unsupervised machine learning
   $env:NUMBA_PRECOMPILE="True"             #Enable precompilation of Numba-based statistical methods. Results in slower SimBA load time but removed runtime cost associated with the first iteration run of any Numba decorated functions.
   $env:CUML="True"                         #Enables GUI access to methods fitting supervised machine learning models using GPU device


Visualizations crash with an [xcb] error on Linux
--------------------------------------------------

On some Linux systems, creating visualizations (e.g., classification videos, gantt plots, heatmaps) stalls, and the terminal repeatedly prints:

.. code-block:: text

    [xcb] Unknown sequence number while processing queue
    [xcb] You called XInitThreads, this is not your fault
    [xcb] Aborting, sorry about that.

This typically happens with remote or virtual displays, such as HPC desktops (e.g., Open OnDemand), VNC, and SSH X-forwarding. By default on Linux, SimBA starts its worker processes with ``fork``, and the workers inherit the SimBA GUI's connection to the display. To start the workers with ``spawn`` instead, set:

.. code-block:: bash

    export MP_START_METHOD=spawn
    simba

On a shared system, such as an HPC, the ``export`` line can be added to the SimBA module file or environment, so that users only need to run ``simba``.

``MP_START_METHOD`` accepts ``fork``, ``spawn`` and ``forkserver`` (on Windows, only ``spawn`` is available). If not set, SimBA uses the operating system default (``spawn`` on Windows, macOS and WSL; ``fork`` on other Linux systems). With ``spawn``, the workers take slightly longer to start, as each worker loads SimBA.










