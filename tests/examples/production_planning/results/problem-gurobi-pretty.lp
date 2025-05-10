\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x92be20112224ce2e
Maximize
  5 Production[1] + 3.5 Production[2] + 4.5 Production[3]
Subject To
 machine_capacity[1]: - 540 ONE + 3 Production[1] + 5 Production[2]
   + 4 Production[3] <= 0
 machine_capacity[2]: - 480 ONE + 6 Production[1] + Production[2]
   + 3 Production[3] <= 0
Bounds
 ONE = 1
End
