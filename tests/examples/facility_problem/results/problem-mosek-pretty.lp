\ Created by MOSEK version 10.2.19
minimize
  + 0 ONE + 12000 open_0 + 15000 open_1 + 17000 open_2 + 13000 open_3 + 16000 open_4
 + 4000 transport_0_0 + 2000 transport_0_1 + 3000 transport_0_2 + 2500 transport_0_3
 + 4500 transport_0_4 + 2500 transport_1_0 + 2600 transport_1_1 + 3400 transport_1_2
 + 3000 transport_1_3 + 4000 transport_1_4 + 1200 transport_2_0 + 1800 transport_2_1
 + 2600 transport_2_2 + 41400 transport_2_3 + 3000 transport_2_4 + 2200 transport_3_0
 + 2600 transport_3_1 + 3100 transport_3_2 + 3700 transport_3_3 + 3200 transport_3_4
subject to
 con_max_capacity_0:  - 20 open_0 + transport_0_0 + transport_1_0 + transport_2_0
 + transport_3_0 <= 0
 con_max_capacity_1:  - 22 open_1 + transport_0_1 + transport_1_1 + transport_2_1
 + transport_3_1 <= 0
 con_max_capacity_2:  - 17 open_2 + transport_0_2 + transport_1_2 + transport_2_2
 + transport_3_2 <= 0
 con_max_capacity_3:  - 19 open_3 + transport_0_3 + transport_1_3 + transport_2_3
 + transport_3_3 <= 0
 con_max_capacity_4:  - 18 open_4 + transport_0_4 + transport_1_4 + transport_2_4
 + transport_3_4 <= 0
 con_meet_demand_0:  - 15 ONE + transport_0_0 + transport_0_1 + transport_0_2 + transport_0_3
 + transport_0_4 = 0
 con_meet_demand_1:  - 18 ONE + transport_1_0 + transport_1_1 + transport_1_2 + transport_1_3
 + transport_1_4 = 0
 con_meet_demand_2:  - 14 ONE + transport_2_0 + transport_2_1 + transport_2_2 + transport_2_3
 + transport_2_4 = 0
 con_meet_demand_3:  - 20 ONE + transport_3_0 + transport_3_1 + transport_3_2 + transport_3_3
 + transport_3_4 = 0
bounds
 1 <= ONE <= 1
 0 <= open_0 <= 1
 0 <= open_1 <= 1
 0 <= open_2 <= 1
 0 <= open_3 <= 1
 0 <= open_4 <= 1
 0 <= transport_0_0 <= +infinity
 0 <= transport_0_1 <= +infinity
 0 <= transport_0_2 <= +infinity
 0 <= transport_0_3 <= +infinity
 0 <= transport_0_4 <= +infinity
 0 <= transport_1_0 <= +infinity
 0 <= transport_1_1 <= +infinity
 0 <= transport_1_2 <= +infinity
 0 <= transport_1_3 <= +infinity
 0 <= transport_1_4 <= +infinity
 0 <= transport_2_0 <= +infinity
 0 <= transport_2_1 <= +infinity
 0 <= transport_2_2 <= +infinity
 0 <= transport_2_3 <= +infinity
 0 <= transport_2_4 <= +infinity
 0 <= transport_3_0 <= +infinity
 0 <= transport_3_1 <= +infinity
 0 <= transport_3_2 <= +infinity
 0 <= transport_3_3 <= +infinity
 0 <= transport_3_4 <= +infinity
general
  open_0 open_1 open_2 open_3 open_4
end
