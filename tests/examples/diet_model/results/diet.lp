minimize

obj:

+1.5 Buy[hot_dog] +2.49 Buy[hamburger] +2.49 Buy[salad] +1.99 Buy[pizza] +0.89 Buy[milk] +1.89 Buy[fries] +2.09 Buy[macaroni] +1.59 Buy[ice_cream] +2.89 Buy[chicken] 

s.t.

min_nutrients[calories]: -560.0 Buy[hot_dog] -410.0 Buy[hamburger] -320.0 Buy[salad] -320.0 Buy[pizza] -100.0 Buy[milk] -380.0 Buy[fries] -320.0 Buy[macaroni] -330.0 Buy[ice_cream] -420.0 Buy[chicken]  <= -1800.0
min_nutrients[protein]: -20.0 Buy[hot_dog] -24.0 Buy[hamburger] -31.0 Buy[salad] -15.0 Buy[pizza] -8.0 Buy[milk] -4.0 Buy[fries] -12.0 Buy[macaroni] -8.0 Buy[ice_cream] -32.0 Buy[chicken]  <= -91.0
max_nutrients[calories]: +560.0 Buy[hot_dog] +410.0 Buy[hamburger] +320.0 Buy[salad] +320.0 Buy[pizza] +100.0 Buy[milk] +380.0 Buy[fries] +320.0 Buy[macaroni] +330.0 Buy[ice_cream] +420.0 Buy[chicken]  <= 2200.0
max_nutrients[fat]: +32.0 Buy[hot_dog] +26.0 Buy[hamburger] +12.0 Buy[salad] +12.0 Buy[pizza] +2.5 Buy[milk] +19.0 Buy[fries] +10.0 Buy[macaroni] +10.0 Buy[ice_cream] +10.0 Buy[chicken]  <= 65.0
max_nutrients[sodium]: +1800.0 Buy[hot_dog] +730.0 Buy[hamburger] +1230.0 Buy[salad] +820.0 Buy[pizza] +125.0 Buy[milk] +270.0 Buy[fries] +930.0 Buy[macaroni] +180.0 Buy[ice_cream] +1190.0 Buy[chicken]  <= 1779.0


bounds

+0 <= Buy[hot_dog] <= +inf
+0 <= Buy[hamburger] <= +inf
+0 <= Buy[salad] <= +inf
+0 <= Buy[pizza] <= +inf
+0 <= Buy[milk] <= +inf
+0 <= Buy[fries] <= +inf
+0 <= Buy[macaroni] <= +inf
+0 <= Buy[ice_cream] <= +inf
+0 <= Buy[chicken] <= +inf
end
