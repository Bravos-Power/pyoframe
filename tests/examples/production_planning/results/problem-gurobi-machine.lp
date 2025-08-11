\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x92be20112224ce2e
Maximize
  5 V + 3.5 V + 4.5 V
Subject To
 C: - 540 ONE + 3 V + 5 V + 4 V <= 0
 C: - 480 ONE + 6 V + V + 3 V <= 0
Bounds
 ONE = 1
End
