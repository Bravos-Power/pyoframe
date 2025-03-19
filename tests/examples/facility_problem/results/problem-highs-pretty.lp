\ File written by HiGHS .lp file handler
min
 obj: +12000 open[0] +15000 open[1] +17000 open[2] +13000 open[3] +16000 open[4] +4000 transport[0,0] +2000 transport[0,1] +3000 transport[0,2] +2500 transport[0,3] +4500 transport[0,4] +2500 transport[1,0] +2600 transport[1,1] +3400 transport[1,2] +3000 transport[1,3] +4000 transport[1,4] +1200 transport[2,0] +1800 transport[2,1] +2600 transport[2,2] +41400 transport[2,3] +3000 transport[2,4] +2200 transport[3,0] +2600 transport[3,1] +3100 transport[3,2] +3700 transport[3,3] +3200 transport[3,4]
st
 con_max_capacity[0]: -20 open[0] +1 transport[0,0] +1 transport[1,0] +1 transport[2,0] +1 transport[3,0] <= +0
 con_max_capacity[1]: -22 open[1] +1 transport[0,1] +1 transport[1,1] +1 transport[2,1] +1 transport[3,1] <= +0
 con_max_capacity[2]: -17 open[2] +1 transport[0,2] +1 transport[1,2] +1 transport[2,2] +1 transport[3,2] <= +0
 con_max_capacity[3]: -19 open[3] +1 transport[0,3] +1 transport[1,3] +1 transport[2,3] +1 transport[3,3] <= +0
 con_max_capacity[4]: -18 open[4] +1 transport[0,4] +1 transport[1,4] +1 transport[2,4] +1 transport[3,4] <= +0
 con_meet_demand[0]: -15 ONE +1 transport[0,0] +1 transport[0,1] +1 transport[0,2] +1 transport[0,3] +1 transport[0,4] = +0
 con_meet_demand[1]: -18 ONE +1 transport[1,0] +1 transport[1,1] +1 transport[1,2] +1 transport[1,3] +1 transport[1,4] = +0
 con_meet_demand[2]: -14 ONE +1 transport[2,0] +1 transport[2,1] +1 transport[2,2] +1 transport[2,3] +1 transport[2,4] = +0
 con_meet_demand[3]: -20 ONE +1 transport[3,0] +1 transport[3,1] +1 transport[3,2] +1 transport[3,3] +1 transport[3,4] = +0
bounds
 ONE = 1
 open[0] <= 1
 open[1] <= 1
 open[2] <= 1
 open[3] <= 1
 open[4] <= 1
bin
 open[0]
 open[1]
 open[2]
 open[3]
 open[4]
gen
semi
end
