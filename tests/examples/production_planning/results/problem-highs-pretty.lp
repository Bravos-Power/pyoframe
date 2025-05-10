\ File written by HiGHS .lp file handler
max
 obj: +5 Production[1] +3.5 Production[2] +4.5 Production[3]
st
 machine_capacity[1]: -540 ONE +3 Production[1] +5 Production[2] +4 Production[3] <= +0
 machine_capacity[2]: -480 ONE +6 Production[1] +1 Production[2] +3 Production[3] <= +0
bounds
 ONE = 1
end
