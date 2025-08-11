\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xfc143aea12ce2404
Minimize
 [ 0.02 V ^2 + 0.02 V * V + 0.008 V * V + 0.004 V * V + 0.002 V * V
   + 0.08 V ^2 + 0.04 V * V + 0.02 V * V + 0.008 V * V + 0.18 V ^2
   + 0.08 V * V + 0.04 V * V + 0.32 V ^2 + 0.16 V * V + 0.5 V ^2
   ] / 2 
Subject To
 C: - ONE + V + V + V + V + V = 0
 C: - 0.1 ONE + 0.05 V + 0.07 V + 0.12 V + 0.15 V + 0.18 V >= 0
Bounds
 ONE = 1
 V <= 0.5
 V <= 0.5
 V <= 0.5
 V <= 0.5
 V <= 0.5
End
