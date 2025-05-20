# Unit Commitment Problem: Pumped-storage hydroelectricity

Problem statement:

> Every hour, a [pumped hydro plant](https://en.wikipedia.org/wiki/Pumped-storage_hydroelectricity) can choose to _either_ pump water uphill or generate power from its turbine by letting water run downhill. At the start of the year, the reservoir contains 300 MWh of energy. At the end of the year, it must return to this amount (such that the model is more realistic of the constraints of multi-year operation). The reservoir can never go below 100 MWh and its maximal energy storage capacity is 630 MWh. When the pump is on, it draws 70 MW of power and has a 75% efficiency. The turbine can output anywhere between 0 and 90 MW of power (when the pump is not in use). Given the cost of electricity at every hour, find the pump and turbine "schedule" that would maximize operational profits (turbine energy sold - pump energy bought).

The input data is using real electricity prices from the German and Luxembourg market stored in `elspot-prices_2021_hourly_eur.csv`.

