\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2.49 Buy[veggieburger] + 2.89 Buy[tofu] + 1.5 Buy[lentils]
   + 1.89 Buy[chickpeas] + 2.09 Buy[macaroni] + 1.99 Buy[pizza]
   + 2.49 Buy[salad] + 0.89 Buy[milk] + 1.59 Buy[ice_cream]
Subject To
 Buy_ub[veggieburger]: - 40 ONE + Buy[veggieburger] <= 0
 Buy_ub[tofu]: - 30 ONE + Buy[tofu] <= 0
 Buy_ub[lentils]: - 20 ONE + Buy[lentils] <= 0
 Buy_ub[chickpeas]: - 20 ONE + Buy[chickpeas] <= 0
 Buy_ub[macaroni]: - 40 ONE + Buy[macaroni] <= 0
 Buy_ub[pizza]: - 50 ONE + Buy[pizza] <= 0
 Buy_ub[salad]: - 60 ONE + Buy[salad] <= 0
 Buy_ub[milk]: - 6.8 ONE + Buy[milk] <= 0
 Buy_ub[ice_cream]: - 80 ONE + Buy[ice_cream] <= 0
 min_nutrients[calories]: 1800 ONE - 410 Buy[veggieburger] - 420 Buy[tofu]
   - 560 Buy[lentils] - 380 Buy[chickpeas] - 320 Buy[macaroni]
   - 320 Buy[pizza] - 320 Buy[salad] - 100 Buy[milk] - 330 Buy[ice_cream]
   <= 0
 min_nutrients[protein]: 91 ONE - 24 Buy[veggieburger] - 32 Buy[tofu]
   - 20 Buy[lentils] - 4 Buy[chickpeas] - 12 Buy[macaroni] - 15 Buy[pizza]
   - 31 Buy[salad] - 8 Buy[milk] - 8 Buy[ice_cream] <= 0
 max_nutrients[calories]: - 2200 ONE + 410 Buy[veggieburger]
   + 420 Buy[tofu] + 560 Buy[lentils] + 380 Buy[chickpeas]
   + 320 Buy[macaroni] + 320 Buy[pizza] + 320 Buy[salad] + 100 Buy[milk]
   + 330 Buy[ice_cream] <= 0
 max_nutrients[fat]: - 65 ONE + 26 Buy[veggieburger] + 10 Buy[tofu]
   + 32 Buy[lentils] + 19 Buy[chickpeas] + 10 Buy[macaroni] + 12 Buy[pizza]
   + 12 Buy[salad] + 2.5 Buy[milk] + 10 Buy[ice_cream] <= 0
 max_nutrients[sodium]: - 1779 ONE + 730 Buy[veggieburger] + 1190 Buy[tofu]
   + 1800 Buy[lentils] + 270 Buy[chickpeas] + 930 Buy[macaroni]
   + 820 Buy[pizza] + 1230 Buy[salad] + 125 Buy[milk] + 180 Buy[ice_cream]
   <= 0
Bounds
 ONE = 1
End
