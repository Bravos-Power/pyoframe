\ File written by HiGHS .lp file handler
min
 obj: + [ +0.02 c_ekk0 * c_ekk0 +0.02 c_ekk0 * c_ekk1 +0.008 c_ekk0 * c_ekk2 +0.004 c_ekk0 * c_ekk3 +0.002 c_ekk0 * c_ekk4 +0.08 c_ekk1 * c_ekk1 +0.04 c_ekk1 * c_ekk2 +0.02 c_ekk1 * c_ekk3 +0.008 c_ekk1 * c_ekk4 +0.18 c_ekk2 * c_ekk2 +0.08 c_ekk2 * c_ekk3 +0.04 c_ekk2 * c_ekk4 +0.32 c_ekk3 * c_ekk3 +0.16 c_ekk3 * c_ekk4 +0.5 c_ekk4 * c_ekk4  ]/2 
st
 r0: -1 ONE +1 c_ekk0 +1 c_ekk1 +1 c_ekk2 +1 c_ekk3 +1 c_ekk4 = +0
 r1: -0.1 ONE +0.05 c_ekk0 +0.07 c_ekk1 +0.12 c_ekk2 +0.15 c_ekk3 +0.18 c_ekk4 >= +0
bounds
 ONE = 1
 c_ekk0 <= 0.5
 c_ekk1 <= 0.5
 c_ekk2 <= 0.5
 c_ekk3 <= 0.5
 c_ekk4 <= 0.5
end
