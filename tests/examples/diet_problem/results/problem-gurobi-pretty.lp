\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x286caa6ff8905913
Minimize
  2.49 Buy[hamburger] + 2.89 Buy[chicken] + 1.5 Buy[hot_dog]
   + 1.89 Buy[fries] + 2.09 Buy[macaroni] + 1.99 Buy[pizza]
   + 2.49 Buy[salad] + 0.89 Buy[milk] + 1.59 Buy[ice_cream]
Subject To
 Buy_ub[hamburger]: - 40 ONE + Buy[hamburger] <= 0
 Buy_ub[chicken]: - 30 ONE + Buy[chicken] <= 0
 Buy_ub[hot_dog]: - 20 ONE + Buy[hot_dog] <= 0
 Buy_ub[fries]: - 20 ONE + Buy[fries] <= 0
 Buy_ub[macaroni]: - 40 ONE + Buy[macaroni] <= 0
 Buy_ub[pizza]: - 50 ONE + Buy[pizza] <= 0
 Buy_ub[salad]: - 60 ONE + Buy[salad] <= 0
 Buy_ub[milk]: - 6.8 ONE + Buy[milk] <= 0
 Buy_ub[ice_cream]: - 80 ONE + Buy[ice_cream] <= 0
 min_nutrients[calories]: 1800 ONE - 410 Buy[hamburger] - 420 Buy[chicken]
   - 560 Buy[hot_dog] - 380 Buy[fries] - 320 Buy[macaroni] - 320 Buy[pizza]
   - 320 Buy[salad] - 100 Buy[milk] - 330 Buy[ice_cream] <= 0
 min_nutrients[protein]: 91 ONE - 24 Buy[hamburger] - 32 Buy[chicken]
   - 20 Buy[hot_dog] - 4 Buy[fries] - 12 Buy[macaroni] - 15 Buy[pizza]
   - 31 Buy[salad] - 8 Buy[milk] - 8 Buy[ice_cream] <= 0
 max_nutrients[calories]: - 2200 ONE + 410 Buy[hamburger]
   + 420 Buy[chicken] + 560 Buy[hot_dog] + 380 Buy[fries]
   + 320 Buy[macaroni] + 320 Buy[pizza] + 320 Buy[salad] + 100 Buy[milk]
   + 330 Buy[ice_cream] <= 0
 max_nutrients[fat]: - 65 ONE + 26 Buy[hamburger] + 10 Buy[chicken]
   + 32 Buy[hot_dog] + 19 Buy[fries] + 10 Buy[macaroni] + 12 Buy[pizza]
   + 12 Buy[salad] + 2.5 Buy[milk] + 10 Buy[ice_cream] <= 0
 max_nutrients[sodium]: - 1779 ONE + 730 Buy[hamburger] + 1190 Buy[chicken]
   + 1800 Buy[hot_dog] + 270 Buy[fries] + 930 Buy[macaroni]
   + 820 Buy[pizza] + 1230 Buy[salad] + 125 Buy[milk] + 180 Buy[ice_cream]
   <= 0
Bounds
 ONE = 1
End
