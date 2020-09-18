***************
Getting Started
***************

Manual Mode
===========

To execute in **manual mode**, from root folder, type:
``pipenv run python -m alphaDeesp.main -l 9 -s 0 -c 0 -t 0``

--ltc | -l int
                            Integer representing the line to cut.
                            For the moment, only one line to cut is handled
--snapshot | -s int
                            If 1, will generate plots of the different grid topologies
                            managed by alphadeesp and store it in alphadeesp/ressources/output
--chronicscenario | -c int
                            Integer representing the chronic scenario to consider, starting from 0.
                            By default, the first available chronic scenario will be chosen, i.e. argument is 0
--timestep | -t int
                            Integer representing the timestep number at
                            which we want to run alphadeesp simulation

In any case, an end result dataframe is written in root folder

In manual mode, further configuration is made through alphadeesp/config.ini

* *simulatorType* - you can chose Grid2op or Pypownet
* *gridPath* - path to folder containing files representing the grid
* *CustomLayout* - list of couples reprenting coordinates of grid nodes. If not provided, grid2op will load grid_layout.json in grid folder
* *grid2opDifficulty* - "0", "1", "2" or "competition". Be careful: grid datasets should have a difficulty_levels.json
* *7 other constants for alphadeesp computation* can be set in config.ini, with comments within the file


Agent Mode
==========

To execute in **agent mode**, please refer to ExpertAgent available in l2rpn-baseline repository

https://github.com/mjothy/l2rpn-baselines/tree/mj-devs/l2rpn_baselines/ExpertAgent

Instead of configuring through config.ini, you can pass a similar python dictionary to the API


Tests
=====

To launch the test suite:
`pipenv run python -m pytest --verbose --continue-on-collection-errors -p no:warnings
`

Debug Help
==========
- To force specific hubs
in AlphaDeesp.compute_best_topo() function, one can force override the hubs result. Check in code, there are
commented examples.

- To force specific combinations for hubs
If one wants a specific hub, a user can "force" a specific node combination.
Check in the code, there are commented examples
