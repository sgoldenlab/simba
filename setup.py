"""
SimBA (Simple Behavioral Analysis)
https://github.com/sgoldenlab/simba
Contributors.
https://github.com/sgoldenlab/simba#contributors-
Licensed under GNU Lesser General Public License v3.0
"""

import setuptools

setuptools.setup(
    name="Simba-UW-tf-dev",
    version="0.91.8",
    author="Simon Nilsson, Jia Jie Choong, Nastacia Goodwin, Sophia Hwang, Sam Golden",
    author_email="goldenneurolab@gmail.com",
    description="Toolkit for computer classification of complex social behaviors in experimental animals",
    url="https://github.com/sgoldenlab/simba",
    install_requires=['Pillow == 5.4.1', 'pyyaml == 5.3.1','shapely == 1.7','wxpython == 4.0.4',
              'dtreeviz == 0.8.1','eli5 == 0.10.1','graphviz == 0.11',
              'imblearn == 0.0','imgaug == 0.4.0','imutils == 0.5.2','matplotlib == 3.0.3', 'numpy == 1.22.0',
              'opencv-python == 3.4.5.20','pandas == 0.25.3','scikit-image == 0.14.2', 'scipy == 1.1.0','seaborn == 0.9.0',
              'scikit-learn == 0.22.2','tabulate == 0.8.3','tqdm == 4.30.0','xgboost == 0.90',
              'yellowbrick == 0.9.1', 'dash == 1.14.0', 'dash_color_picker == 0.0.1', 'dash_core_components == 1.10.2', 
			  'dash_colorscales == 0.0.4', 'dash_html_components == 1.0.3', 'h5py == 2.9.0', 'numba == 0.48.0', 'numexpr == 2.6.9',
			  'plotly == 4.9.0', 'statsmodels == 0.9.0', 'cefpython3 == 66.0', "pyarrow == 0.17.1", "shap == 0.35.0","h5py==2.9.0","tables==3.6.1", "xlrd==1.2.0"],
    license='GNU Lesser General Public License v3 (LGPLv3)',
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points={'console_scripts':['simba=simba.SimBA:main'],}
)
