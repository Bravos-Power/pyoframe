minimize

obj:

+1.59 Buy[ice_cream] +2.89 Buy[chicken] +0.89 Buy[milk] +1.99 Buy[pizza] +1.89 Buy[fries] +1.5 Buy[hot_dog] +2.49 Buy[salad] +2.09 Buy[macaroni] +2.49 Buy[hamburger] 

s.t.

min_nutrients[calories]: -330.0 Buy[ice_cream] -420.0 Buy[chicken] -100.0 Buy[milk] -320.0 Buy[pizza] -380.0 Buy[fries] -560.0 Buy[hot_dog] -320.0 Buy[salad] -320.0 Buy[macaroni] -410.0 Buy[hamburger]  <= -1800.0
min_nutrients[protein]: -8.0 Buy[ice_cream] -32.0 Buy[chicken] -8.0 Buy[milk] -15.0 Buy[pizza] -4.0 Buy[fries] -20.0 Buy[hot_dog] -31.0 Buy[salad] -12.0 Buy[macaroni] -24.0 Buy[hamburger]  <= -91.0
max_nutrients[calories]: +330.0 Buy[ice_cream] +420.0 Buy[chicken] +100.0 Buy[milk] +320.0 Buy[pizza] +380.0 Buy[fries] +560.0 Buy[hot_dog] +320.0 Buy[salad] +320.0 Buy[macaroni] +410.0 Buy[hamburger]  <= 2200.0
max_nutrients[fat]: +10.0 Buy[ice_cream] +10.0 Buy[chicken] +2.5 Buy[milk] +12.0 Buy[pizza] +19.0 Buy[fries] +32.0 Buy[hot_dog] +12.0 Buy[salad] +10.0 Buy[macaroni] +26.0 Buy[hamburger]  <= 65.0
max_nutrients[sodium]: +180.0 Buy[ice_cream] +1190.0 Buy[chicken] +125.0 Buy[milk] +820.0 Buy[pizza] +270.0 Buy[fries] +1800.0 Buy[hot_dog] +1230.0 Buy[salad] +930.0 Buy[macaroni] +730.0 Buy[hamburger]  <= 1779.0


bounds

+0 <= Buy[ice_cream] <= +inf
+0 <= Buy[chicken] <= +inf
+0 <= Buy[milk] <= +inf
+0 <= Buy[pizza] <= +inf
+0 <= Buy[fries] <= +inf
+0 <= Buy[hot_dog] <= +inf
+0 <= Buy[salad] <= +inf
+0 <= Buy[macaroni] <= +inf
+0 <= Buy[hamburger] <= +inf
end
