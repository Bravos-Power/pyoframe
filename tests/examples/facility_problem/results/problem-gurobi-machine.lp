\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x13f1dc31c723ad33
Minimize
  12000 C1 + 15000 C2 + 17000 C3 + 13000 C4 + 16000 C5 + 4000 C6 + 2000 C7
   + 3000 C8 + 2500 C9 + 4500 C10 + 2500 C11 + 2600 C12 + 3400 C13
   + 3000 C14 + 4000 C15 + 1200 C16 + 1800 C17 + 2600 C18 + 41400 C19
   + 3000 C20 + 2200 C21 + 2600 C22 + 3100 C23 + 3700 C24 + 3200 C25
Subject To
 R0: - 20 C1 + C6 + C11 + C16 + C21 <= 0
 R1: - 22 C2 + C7 + C12 + C17 + C22 <= 0
 R2: - 17 C3 + C8 + C13 + C18 + C23 <= 0
 R3: - 19 C4 + C9 + C14 + C19 + C24 <= 0
 R4: - 18 C5 + C10 + C15 + C20 + C25 <= 0
 R5: - 15 ONE + C6 + C7 + C8 + C9 + C10 = 0
 R6: - 18 ONE + C11 + C12 + C13 + C14 + C15 = 0
 R7: - 14 ONE + C16 + C17 + C18 + C19 + C20 = 0
 R8: - 20 ONE + C21 + C22 + C23 + C24 + C25 = 0
Bounds
 ONE = 1
Binaries
 C1 C2 C3 C4 C5
End
