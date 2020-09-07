pipenv shell
pip install pyinstrument
pip install gprof2dot pour avoir des graphes (sinon commenter la ligne de code qui y fait référence dans profiling_alphadeespp.py)


CHOOSE NETWORK AND CONF
In alphadeesp_process.py
In network l2rpn_2020_wcci, change thermal limit of line 13 (to 70A for instance) in order to trigger overflows

LAUNCH REPORT ON WHOLE EXECUTION (impossible if infinite execution)
pyinstrument -r html alphadeesp_process.py

LAUNCH REPORT ON TIME LIMITED RUN
Set number of seconds in profiling_alphadeesp.py
python profiling_alphadeesp.py

