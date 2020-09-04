pipenv shell
pip install pyinstrument
In profiling_alphadeesp.py --> change max number of seconds to be ran (otherwise it is infinite)
pyinstrument -r html profiling_alphadeesp.py