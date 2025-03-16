\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x286caa6ff8905913
Minimize
  2.49 C1 + 2.89 C2 + 1.5 C3 + 1.89 C4 + 2.09 C5 + 1.99 C6 + 2.49 C7
   + 0.89 C8 + 1.59 C9
Subject To
 R0: - 40 ONE + C1 <= 0
 R1: - 30 ONE + C2 <= 0
 R2: - 20 ONE + C3 <= 0
 R3: - 20 ONE + C4 <= 0
 R4: - 40 ONE + C5 <= 0
 R5: - 50 ONE + C6 <= 0
 R6: - 60 ONE + C7 <= 0
 R7: - 6.8 ONE + C8 <= 0
 R8: - 80 ONE + C9 <= 0
 R9: 1800 ONE - 410 C1 - 420 C2 - 560 C3 - 380 C4 - 320 C5 - 320 C6
   - 320 C7 - 100 C8 - 330 C9 <= 0
 R10: 91 ONE - 24 C1 - 32 C2 - 20 C3 - 4 C4 - 12 C5 - 15 C6 - 31 C7 - 8 C8
   - 8 C9 <= 0
 R11: - 2200 ONE + 410 C1 + 420 C2 + 560 C3 + 380 C4 + 320 C5 + 320 C6
   + 320 C7 + 100 C8 + 330 C9 <= 0
 R12: - 65 ONE + 26 C1 + 10 C2 + 32 C3 + 19 C4 + 10 C5 + 12 C6 + 12 C7
   + 2.5 C8 + 10 C9 <= 0
 R13: - 1779 ONE + 730 C1 + 1190 C2 + 1800 C3 + 270 C4 + 930 C5 + 820 C6
   + 1230 C7 + 125 C8 + 180 C9 <= 0
Bounds
 ONE = 1
End
