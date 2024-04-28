\ Model facility
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  12000 open[0] + 15000 open[1] + 17000 open[2] + 13000 open[3]
   + 16000 open[4] + 4000 transport[0,0] + 2000 transport[0,1]
   + 3000 transport[0,2] + 2500 transport[0,3] + 4500 transport[0,4]
   + 2500 transport[1,0] + 2600 transport[1,1] + 3400 transport[1,2]
   + 3000 transport[1,3] + 4000 transport[1,4] + 1200 transport[2,0]
   + 1800 transport[2,1] + 2600 transport[2,2] + 41400 transport[2,3]
   + 3000 transport[2,4] + 2200 transport[3,0] + 2600 transport[3,1]
   + 3100 transport[3,2] + 3700 transport[3,3] + 3200 transport[3,4]
Subject To
 Capacity[0]: - 20 open[0] + transport[0,0] + transport[1,0]
   + transport[2,0] + transport[3,0] <= 0
 Capacity[1]: - 22 open[1] + transport[0,1] + transport[1,1]
   + transport[2,1] + transport[3,1] <= 0
 Capacity[2]: - 17 open[2] + transport[0,2] + transport[1,2]
   + transport[2,2] + transport[3,2] <= 0
 Capacity[3]: - 19 open[3] + transport[0,3] + transport[1,3]
   + transport[2,3] + transport[3,3] <= 0
 Capacity[4]: - 18 open[4] + transport[0,4] + transport[1,4]
   + transport[2,4] + transport[3,4] <= 0
 Demand[0]: transport[0,0] + transport[0,1] + transport[0,2]
   + transport[0,3] + transport[0,4] = 15
 Demand[1]: transport[1,0] + transport[1,1] + transport[1,2]
   + transport[1,3] + transport[1,4] = 18
 Demand[2]: transport[2,0] + transport[2,1] + transport[2,2]
   + transport[2,3] + transport[2,4] = 14
 Demand[3]: transport[3,0] + transport[3,1] + transport[3,2]
   + transport[3,3] + transport[3,4] = 20
Bounds
Binaries
 open[0] open[1] open[2] open[3] open[4]
End
