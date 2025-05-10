\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x92be20112224ce2e
Maximize
  5 C1 + 3.5 C2 + 4.5 C3
Subject To
 R0: - 540 ONE + 3 C1 + 5 C2 + 4 C3 <= 0
 R1: - 480 ONE + 6 C1 + C2 + 3 C3 <= 0
Bounds
 ONE = 1
End
