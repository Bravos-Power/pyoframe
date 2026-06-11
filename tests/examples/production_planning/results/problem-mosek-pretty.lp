\ Created by MOSEK version 10.2.19
maximize
  + 0 ONE + 5 Production_1 + 3.5 Production_2 + 4.5 Production_3
subject to
 machine_capacity_1:  - 540 ONE + 3 Production_1 + 5 Production_2 + 4 Production_3 <= 0
 machine_capacity_2:  - 480 ONE + 6 Production_1 + Production_2 + 3 Production_3 <= 0
bounds
 1 <= ONE <= 1
 0 <= Production_1 <= +infinity
 0 <= Production_2 <= +infinity
 0 <= Production_3 <= +infinity
end
