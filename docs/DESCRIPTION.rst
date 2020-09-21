***********
Description
***********

Introduction
============

This module represents an expert agent that finds solutions to optimize a power network. The expert agent is based
on a research paper: (link)

The concept is, given a line in overflow (referred as *Line to cut*) the expert agent will run simulations on the network
and try to find and rank the best topological actions (changing elements of the graph from one bus to the other) to solve
the overflow.

Important limitations
=====================

- For the moment, we allow cutting only one line when launching the expert system:
    * ex python3 -m alphaDeesp.main -l 9

- The agent will only take the given timestep into account, meaning it will not try to learn from past or future behavior

- **Pypownet only** Only works with initial state of all nodes with busbar == 0

- **Pypownet only** At the moment, in the internal computation, a substation can have only one source of Power and one source of Consumption

Data Flow
=========

The Data flow starts with an Observation object, whether from Pypownet or Grid2op API. This Observation object will set the timestep of all simulations.

- First, you load the Observation: simulation.load(Observation)
- Then, it **creates a dictionnary, self.topo**:
    * self.topo["edges"]["idx_or"] = [x for x in idx_or]
    * self.topo["edges"]["idx_ex"] = [x for x in idx_ex]
    * self.topo["edges"]["init_flows"] = current_flows
    * self.topo["nodes"]["are_prods"] = are_prods
    * self.topo["nodes"]["are_loads"] = are_loads
    * self.topo["nodes"]["prods_values"] = prods_values
    * self.topo["nodes"]["loads_values"] = loads_values
- From self.topo dict, a **DataFrame is created**: self.df with column indices being as such (idx_or  idx_ex  init_flows
  swapped  new_flows new_flows_swapped  delta_flows  gray_edges) and row indices being the lines IDs in
- creates and fill internal structures from DataFrame

Explanation
===========

Before heading into a brief explanation of the algorithm

There are three important objects to have in mind:

* g_pow - A powerflow_graph: it displays electricity flow on edges (in MW).

.. image:: ../alphaDeesp/ressources/g_pow_print.jpg

* g_pow_prime - A powerflow_graph: it displays the electricity flow after a line has been cut, here in the example we
can see the line n°9 that has been cut, it now has a value of 0

.. image:: ../alphaDeesp/ressources/g_pow_prime_print.jpg

* g_over - An Overflow graph: it is the result of "g_pow" that got compared to "g_pow_prime". The edge's values represent the difference between g_pow_prime_edge_value - g_pow_edge_value

**g_over = g_pow_prime - g_pow**

.. image::  ../alphaDeesp/ressources/g_over_print.jpg

Now, to the main algorithm. The first three steps of the algorithm are about extracting the situation, creating and
structuring the data that will be needed for the rest of the steps.

.. image::  ../alphaDeesp/ressources/first_line_algorithm_es_.png

At this step there is a Overload Graph coupled with organized data in a Dataframe that will enable to do the rest of the steps.
AlphaDeesp needs a NetworkX graph, a DataFrame, and another dictionary with specific data to properly work.

.. image::  ../alphaDeesp/ressources/second_line_algorithm_es_.png

Now all substations are ranked with our expert knowledge, the last steps consist of simulating the top N
(can be changed in config.ini file) topologies with a simulator and rank them accordingly.

.. image:: ../alphaDeesp/ressources/third_line_algorithm_es_.png


explain internal structure, and how another API simulator could be plugged in.

mention at which step you can have a graphical print. (when we can display a graph and where)


Important Information
=====================

If an element (Production, Consumption, OriginLine, ExtremityLine) is on busbar 0 with ID X,
it will appear on the display graph on node X

However, if an element is on busbar 1 with ID X,
The program will create another node named 666X