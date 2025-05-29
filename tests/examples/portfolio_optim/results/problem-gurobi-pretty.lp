\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xfc143aea12ce2404
Minimize
 [ 0.02 weight[A] ^2 + 0.02 weight[A] * weight[B]
   + 0.008 weight[A] * weight[C] + 0.004 weight[A] * weight[D]
   + 0.002 weight[A] * weight[E] + 0.08 weight[B] ^2
   + 0.04 weight[B] * weight[C] + 0.02 weight[B] * weight[D]
   + 0.008 weight[B] * weight[E] + 0.18 weight[C] ^2
   + 0.08 weight[C] * weight[D] + 0.04 weight[C] * weight[E]
   + 0.32 weight[D] ^2 + 0.16 weight[D] * weight[E] + 0.5 weight[E] ^2
   ] / 2 
Subject To
 con_weights_sum: - ONE + weight[A] + weight[B] + weight[C] + weight[D]
   + weight[E] = 0
 con_min_return: - 0.1 ONE + 0.05 weight[A] + 0.07 weight[B]
   + 0.12 weight[C] + 0.15 weight[D] + 0.18 weight[E] >= 0
Bounds
 ONE = 1
 weight[A] <= 0.5
 weight[B] <= 0.5
 weight[C] <= 0.5
 weight[D] <= 0.5
 weight[E] <= 0.5
End
