\ File written by HiGHS .lp file handler
min
 obj: + [ +0.02 col_ekk_0 * col_ekk_0 +0.02 col_ekk_0 * col_ekk_1 +0.008 col_ekk_0 * col_ekk_2 +0.004 col_ekk_0 * col_ekk_3 +0.002 col_ekk_0 * col_ekk_4 +0.08 col_ekk_1 * col_ekk_1 +0.04 col_ekk_1 * col_ekk_2 +0.02 col_ekk_1 * col_ekk_3 +0.008 col_ekk_1 * col_ekk_4 +0.18 col_ekk_2 * col_ekk_2 +0.08 col_ekk_2 * col_ekk_3 +0.04 col_ekk_2 * col_ekk_4 +0.32 col_ekk_3 * col_ekk_3 +0.16 col_ekk_3 * col_ekk_4 +0.5 col_ekk_4 * col_ekk_4  ]/2 
st
 con1: -1 ONE +1 col_ekk_0 +1 col_ekk_1 +1 col_ekk_2 +1 col_ekk_3 +1 col_ekk_4 = +0
 con2: -0.1 ONE +0.05 col_ekk_0 +0.07 col_ekk_1 +0.12 col_ekk_2 +0.15 col_ekk_3 +0.18 col_ekk_4 >= +0
bounds
 ONE = 1
 col_ekk_0 <= 0.5
 col_ekk_1 <= 0.5
 col_ekk_2 <= 0.5
 col_ekk_3 <= 0.5
 col_ekk_4 <= 0.5
end
