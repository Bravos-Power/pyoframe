minimize

obj:

+1.89 Buy[fries] +0.89 Buy[milk] +2.89 Buy[chicken] +1.5 Buy[hot_dog] +2.09 Buy[macaroni] +2.49 Buy[hamburger] +1.99 Buy[pizza] +1.59 Buy[ice_cream] +2.49 Buy[salad] 

s.t.

min_nutrients[calories]: -380.0 Buy[fries] -100.0 Buy[milk] -420.0 Buy[chicken] -560.0 Buy[hot_dog] -320.0 Buy[macaroni] -410.0 Buy[hamburger] -320.0 Buy[pizza] -330.0 Buy[ice_cream] -320.0 Buy[salad]  <= -1800.0
min_nutrients[protein]: -4.0 Buy[fries] -8.0 Buy[milk] -32.0 Buy[chicken] -20.0 Buy[hot_dog] -12.0 Buy[macaroni] -24.0 Buy[hamburger] -15.0 Buy[pizza] -8.0 Buy[ice_cream] -31.0 Buy[salad]  <= -91.0
max_nutrients[calories]: +380.0 Buy[fries] +100.0 Buy[milk] +420.0 Buy[chicken] +560.0 Buy[hot_dog] +320.0 Buy[macaroni] +410.0 Buy[hamburger] +320.0 Buy[pizza] +330.0 Buy[ice_cream] +320.0 Buy[salad]  <= 2200.0
max_nutrients[fat]: +19.0 Buy[fries] +2.5 Buy[milk] +10.0 Buy[chicken] +32.0 Buy[hot_dog] +10.0 Buy[macaroni] +26.0 Buy[hamburger] +12.0 Buy[pizza] +10.0 Buy[ice_cream] +12.0 Buy[salad]  <= 65.0
max_nutrients[sodium]: +270.0 Buy[fries] +125.0 Buy[milk] +1190.0 Buy[chicken] +1800.0 Buy[hot_dog] +930.0 Buy[macaroni] +730.0 Buy[hamburger] +820.0 Buy[pizza] +180.0 Buy[ice_cream] +1230.0 Buy[salad]  <= 1779.0


bounds

+0 <= Buy[fries] <= +inf
+0 <= Buy[milk] <= +inf
+0 <= Buy[chicken] <= +inf
+0 <= Buy[hot_dog] <= +inf
+0 <= Buy[macaroni] <= +inf
+0 <= Buy[hamburger] <= +inf
+0 <= Buy[pizza] <= +inf
+0 <= Buy[ice_cream] <= +inf
+0 <= Buy[salad] <= +inf
end
