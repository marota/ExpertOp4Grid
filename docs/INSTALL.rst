Installation
------------

To install ExpertOp4Grid and AlphaDeesp execute the following lines:


1. (Optional)(Recommended) if you want to run in manual mode, install graphviz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is for neato package, it allows to transform a dot file into a pdf file.

*Warning: It is important to install graphviz executables before python packages*

First install executable

- On Linux

``apt-get install graphviz``

- On Windows, use package finder (equivalent of apt-get on Windows)

``winget install graphviz``

Then ensure that graphviz and neato are in the path. You often have to set it manually. For example on windows you can use the following command line:

``setx /M path "%path%;'C:\Users\username\graphviz-2.38\release\bin"``

Then you can move to python packages installation


2. Install the package from Pypi
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pip install ExpertOp4Grid``


3. (Optional) If you want to run simulation with pypownet instead of Grid2op:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Clone pypownet somewhere else :

``cd ..``
``git clone https://github.com/MarvinLer/pypownet.git``

- Install from within that folder:

``python setup.py install --user``

or

``cd ExpertOp4Grid``
``pipenv shell``
``cd ../pypownet``
``python setup.py install``

4. (Optional) Compile and output the sphinx doc (this documentation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run
``./docs/make.bat html``
