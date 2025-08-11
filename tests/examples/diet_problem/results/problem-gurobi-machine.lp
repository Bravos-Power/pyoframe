\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x286caa6ff8905913
Minimize
  2.49 V + 2.89 V + 1.5 V + 1.89 V + 2.09 V + 1.99 V + 2.49 V
   + 0.89 V + 1.59 V
Subject To
 C: - 40 ONE + V <= 0
 C: - 30 ONE + V <= 0
 C: - 20 ONE + V <= 0
 C: - 20 ONE + V <= 0
 C: - 40 ONE + V <= 0
 C: - 50 ONE + V <= 0
 C: - 60 ONE + V <= 0
 C: - 6.8 ONE + V <= 0
 C: - 80 ONE + V <= 0
 C: 1800 ONE - 410 V - 420 V - 560 V - 380 V - 320 V - 320 V
   - 320 V - 100 V - 330 V <= 0
 C: 91 ONE - 24 V - 32 V - 20 V - 4 V - 12 V - 15 V - 31 V - 8 V
   - 8 V <= 0
 C: - 2200 ONE + 410 V + 420 V + 560 V + 380 V + 320 V + 320 V
   + 320 V + 100 V + 330 V <= 0
 C: - 65 ONE + 26 V + 10 V + 32 V + 19 V + 10 V + 12 V + 12 V
   + 2.5 V + 10 V <= 0
 C: - 1779 ONE + 730 V + 1190 V + 1800 V + 270 V + 930 V + 820 V
   + 1230 V + 125 V + 180 V <= 0
Bounds
 ONE = 1
End
