\ Created by MOSEK version 10.2.19
minimize
  + 0 ONE + 0 weight_A + 0 weight_B + 0 weight_C + 0 weight_D + 0 weight_E + [ + 0.02 weight_A ^ 2
 + 0.02 weight_B * weight_A + 0.008 weight_C * weight_A + 0.004 weight_D * weight_A
 + 0.002 weight_E * weight_A + 0.08 weight_B ^ 2 + 0.04 weight_C * weight_B + 0.02 weight_D * weight_B
 + 0.008 weight_E * weight_B + 0.18 weight_C ^ 2 + 0.08 weight_D * weight_C + 0.04 weight_E * weight_C
 + 0.32 weight_D ^ 2 + 0.16 weight_E * weight_D + 0.5 weight_E ^ 2 ]/2
subject to
 con_weights_sum:  - ONE + weight_A + weight_B + weight_C + weight_D + weight_E = 0
 con_min_return:  - 0.1 ONE + 0.05 weight_A + 0.07 weight_B + 0.12 weight_C + 0.15 weight_D
 + 0.18 weight_E >= 0
bounds
 1 <= ONE <= 1
 0 <= weight_A <= 0.5
 0 <= weight_B <= 0.5
 0 <= weight_C <= 0.5
 0 <= weight_D <= 0.5
 0 <= weight_E <= 0.5
end
