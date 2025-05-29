\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xfc143aea12ce2404
Minimize
 [ 0.02 C1 ^2 + 0.02 C1 * C2 + 0.008 C1 * C3 + 0.004 C1 * C4 + 0.002 C1 * C5
   + 0.08 C2 ^2 + 0.04 C2 * C3 + 0.02 C2 * C4 + 0.008 C2 * C5 + 0.18 C3 ^2
   + 0.08 C3 * C4 + 0.04 C3 * C5 + 0.32 C4 ^2 + 0.16 C4 * C5 + 0.5 C5 ^2
   ] / 2 
Subject To
 R0: - ONE + C1 + C2 + C3 + C4 + C5 = 0
 R1: - 0.1 ONE + 0.05 C1 + 0.07 C2 + 0.12 C3 + 0.15 C4 + 0.18 C5 >= 0
Bounds
 ONE = 1
 C1 <= 0.5
 C2 <= 0.5
 C3 <= 0.5
 C4 <= 0.5
 C5 <= 0.5
End
