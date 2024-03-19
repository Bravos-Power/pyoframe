minimize

obj:

12000 open[0] +15000 open[1] +17000 open[2] +13000 open[3] +16000 open[4] +4000 transport[0,0] +2500 transport[1,0] +1200 transport[2,0] +2200 transport[3,0] +2000 transport[0,1] +2600 transport[1,1] +1800 transport[2,1] +2600 transport[3,1] +3000 transport[0,2] +3400 transport[1,2] +2600 transport[2,2] +3100 transport[3,2] +2500 transport[0,3] +3000 transport[1,3] +41400 transport[2,3] +3700 transport[3,3] +4500 transport[0,4] +4000 transport[1,4] +3000 transport[2,4] +3200 transport[3,4]

s.t.

con_max_capacity[0]: transport[0,0] + transport[1,0] + transport[2,0] + transport[3,0] -20 open[0] <= 0
con_max_capacity[1]: transport[0,1] + transport[1,1] + transport[2,1] + transport[3,1] -22 open[1] <= 0
con_max_capacity[2]: transport[0,2] + transport[1,2] + transport[2,2] + transport[3,2] -17 open[2] <= 0
con_max_capacity[3]: transport[0,3] + transport[1,3] + transport[2,3] + transport[3,3] -19 open[3] <= 0
con_max_capacity[4]: transport[0,4] + transport[1,4] + transport[2,4] + transport[3,4] -18 open[4] <= 0
con_meet_demand[0]: transport[0,0] + transport[0,1] + transport[0,2] + transport[0,3] + transport[0,4] = 15
con_meet_demand[1]: transport[1,0] + transport[1,1] + transport[1,2] + transport[1,3] + transport[1,4] = 18
con_meet_demand[2]: transport[2,0] + transport[2,1] + transport[2,2] + transport[2,3] + transport[2,4] = 14
con_meet_demand[3]: transport[3,0] + transport[3,1] + transport[3,2] + transport[3,3] + transport[3,4] = 20


bounds

-inf <= open[0] <= +inf
-inf <= open[1] <= +inf
-inf <= open[2] <= +inf
-inf <= open[3] <= +inf
-inf <= open[4] <= +inf
+0 <= transport[0,0] <= +inf
+0 <= transport[0,1] <= +inf
+0 <= transport[0,2] <= +inf
+0 <= transport[0,3] <= +inf
+0 <= transport[0,4] <= +inf
+0 <= transport[1,0] <= +inf
+0 <= transport[1,1] <= +inf
+0 <= transport[1,2] <= +inf
+0 <= transport[1,3] <= +inf
+0 <= transport[1,4] <= +inf
+0 <= transport[2,0] <= +inf
+0 <= transport[2,1] <= +inf
+0 <= transport[2,2] <= +inf
+0 <= transport[2,3] <= +inf
+0 <= transport[2,4] <= +inf
+0 <= transport[3,0] <= +inf
+0 <= transport[3,1] <= +inf
+0 <= transport[3,2] <= +inf
+0 <= transport[3,3] <= +inf
+0 <= transport[3,4] <= +inf


binary

open[0]
open[1]
open[2]
open[3]
open[4]
end
