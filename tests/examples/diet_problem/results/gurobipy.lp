\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2.49 Buy[hamburger] + 2.89 Buy[chicken] + 1.5 Buy[hot_dog]
   + 1.89 Buy[fries] + 2.09 Buy[macaroni] + 1.99 Buy[pizza]
   + 2.49 Buy[salad] + 0.89 Buy[milk] + 1.59 Buy[ice_cream]
Subject To
 minNutrition[calories]: 410 Buy[hamburger] + 420 Buy[chicken]
   + 560 Buy[hot_dog] + 380 Buy[fries] + 320 Buy[macaroni] + 320 Buy[pizza]
   + 320 Buy[salad] + 100 Buy[milk] + 330 Buy[ice_cream] >= 1800
 minNutrition[protein]: 24 Buy[hamburger] + 32 Buy[chicken]
   + 20 Buy[hot_dog] + 4 Buy[fries] + 12 Buy[macaroni] + 15 Buy[pizza]
   + 31 Buy[salad] + 8 Buy[milk] + 8 Buy[ice_cream] >= 91
 maxNutrition[calories]: 410 Buy[hamburger] + 420 Buy[chicken]
   + 560 Buy[hot_dog] + 380 Buy[fries] + 320 Buy[macaroni] + 320 Buy[pizza]
   + 320 Buy[salad] + 100 Buy[milk] + 330 Buy[ice_cream] <= 2200
 maxNutrition[fat]: 26 Buy[hamburger] + 10 Buy[chicken] + 32 Buy[hot_dog]
   + 19 Buy[fries] + 10 Buy[macaroni] + 12 Buy[pizza] + 12 Buy[salad]
   + 2.5 Buy[milk] + 10 Buy[ice_cream] <= 65
 maxNutrition[sodium]: 730 Buy[hamburger] + 1190 Buy[chicken]
   + 1800 Buy[hot_dog] + 270 Buy[fries] + 930 Buy[macaroni]
   + 820 Buy[pizza] + 1230 Buy[salad] + 125 Buy[milk] + 180 Buy[ice_cream]
   <= 1779
Bounds
 Buy[hamburger] <= 40
 Buy[chicken] <= 30
 Buy[hot_dog] <= 20
 Buy[fries] <= 20
 Buy[macaroni] <= 40
 Buy[pizza] <= 50
 Buy[salad] <= 60
 Buy[milk] <= 6.8
 Buy[ice_cream] <= 80


End
