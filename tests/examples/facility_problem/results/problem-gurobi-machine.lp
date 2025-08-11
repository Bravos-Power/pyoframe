\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x13f1dc31c723ad33
Minimize
  12000 V + 15000 V + 17000 V + 13000 V + 16000 V + 4000 V + 2000 V
   + 3000 V + 2500 V + 4500 V + 2500 V + 2600 V + 3400 V
   + 3000 V + 4000 V + 1200 V + 1800 V + 2600 V + 41400 V
   + 3000 V + 2200 V + 2600 V + 3100 V + 3700 V + 3200 V
Subject To
 C: - 20 V + V + V + V + V <= 0
 C: - 22 V + V + V + V + V <= 0
 C: - 17 V + V + V + V + V <= 0
 C: - 19 V + V + V + V + V <= 0
 C: - 18 V + V + V + V + V <= 0
 C: - 15 ONE + V + V + V + V + V = 0
 C: - 18 ONE + V + V + V + V + V = 0
 C: - 14 ONE + V + V + V + V + V = 0
 C: - 20 ONE + V + V + V + V + V = 0
Bounds
 ONE = 1
Binaries
 V V V V V
End
