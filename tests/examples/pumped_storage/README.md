# Unit Commitment Problem: Pumped-storage hydroelectricity

This problem is a variant of the unit commitment problem,
which is a fundamental optimization problem in power systems.
The goal is to plan the optimal operation of a [pumped-storage
hydroelectric plant](https://en.wikipedia.org/wiki/Pumped-storage_hydroelectricity) over a given time horizon,
considering various constraints and objectives.
That operation plan is often called "schedules".

Pumped-storage hydroelectric plants are unique in that they can both generate electricity and store it by pumping water
uphill.
Usually, electricity is "pumped-out" in the lake when it is inexpensive and "pumped-in" when it is expensive.

The capacity of the storage lake is limited. In the beginning, the lake has 300 MWh, and at the end of a planning period
it must haveT the same amount of energy stored.
The minimal level of energy of 100 MWh must be maintained in the lake at all times.
The maximal level of energy that can be stored in the lake is 630 MWh

The pumping and generation are per hour granulated, and they are mutually exclusive.
For pumping, the plant can only operate at a capacity of 70 MWh, for generation plant can operate at any capacity from 0
to 90 MWh.
When pumping, there is a 25% lost of the energy; in other words, for every 1 MWh pumped into the lake, only 0.75 MWh is
stored.

The model is a mixed-integer linear programming (MILP) problem,
where the goal is to maximize the profit from electricity generation and storage operations for a year-long time
horizon.
For that we are using real electricity prices from the German and Luxembourg market stored in csv file.

