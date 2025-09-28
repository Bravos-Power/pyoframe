\ File written by HiGHS .lp file handler
min
 obj: + [ +0.02 weight[A] * weight[A] +0.02 weight[A] * weight[B] +0.008 weight[A] * weight[C] +0.004 weight[A] * weight[D] +0.002 weight[A] * weight[E] +0.08 weight[B] * weight[B] +0.04 weight[B] * weight[C] +0.02 weight[B] * weight[D] +0.008 weight[B] * weight[E] +0.18 weight[C] * weight[C] +0.08 weight[C] * weight[D] +0.04 weight[C] * weight[E] +0.32 weight[D] * weight[D] +0.16 weight[D] * weight[E] +0.5 weight[E] * weight[E]  ]/2 
st
 con_weights_sum: -1 ONE +1 weight[A] +1 weight[B] +1 weight[C] +1 weight[D] +1 weight[E] = +0
 con_min_return: -0.1 ONE +0.05 weight[A] +0.07 weight[B] +0.12 weight[C] +0.15 weight[D] +0.18 weight[E] >= +0
bounds
 ONE = 1
 weight[A] <= 0.5
 weight[B] <= 0.5
 weight[C] <= 0.5
 weight[D] <= 0.5
 weight[E] <= 0.5
end
