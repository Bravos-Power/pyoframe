minimize

obj:

+2.89 Buy[chicken] +1.59 Buy[ice_cream] +2.49 Buy[salad] +1.5 Buy[hot_dog] +2.09 Buy[macaroni] +2.49 Buy[hamburger] +0.89 Buy[milk] +1.89 Buy[fries] +1.99 Buy[pizza] 

s.t.

min_nutrients[calories]: -420.0 Buy[chicken] -330.0 Buy[ice_cream] -320.0 Buy[salad] -560.0 Buy[hot_dog] -320.0 Buy[macaroni] -410.0 Buy[hamburger] -100.0 Buy[milk] -380.0 Buy[fries] -320.0 Buy[pizza]  <= -1800.0
min_nutrients[protein]: -32.0 Buy[chicken] -8.0 Buy[ice_cream] -31.0 Buy[salad] -20.0 Buy[hot_dog] -12.0 Buy[macaroni] -24.0 Buy[hamburger] -8.0 Buy[milk] -4.0 Buy[fries] -15.0 Buy[pizza]  <= -91.0
max_nutrients[calories]: +420.0 Buy[chicken] +330.0 Buy[ice_cream] +320.0 Buy[salad] +560.0 Buy[hot_dog] +320.0 Buy[macaroni] +410.0 Buy[hamburger] +100.0 Buy[milk] +380.0 Buy[fries] +320.0 Buy[pizza]  <= 2200.0
max_nutrients[fat]: +10.0 Buy[chicken] +10.0 Buy[ice_cream] +12.0 Buy[salad] +32.0 Buy[hot_dog] +10.0 Buy[macaroni] +26.0 Buy[hamburger] +2.5 Buy[milk] +19.0 Buy[fries] +12.0 Buy[pizza]  <= 65.0
max_nutrients[sodium]: +1190.0 Buy[chicken] +180.0 Buy[ice_cream] +1230.0 Buy[salad] +1800.0 Buy[hot_dog] +930.0 Buy[macaroni] +730.0 Buy[hamburger] +125.0 Buy[milk] +270.0 Buy[fries] +820.0 Buy[pizza]  <= 1779.0


bounds

+0 <= Buy[chicken] <= +inf
+0 <= Buy[ice_cream] <= +inf
+0 <= Buy[salad] <= +inf
+0 <= Buy[hot_dog] <= +inf
+0 <= Buy[macaroni] <= +inf
+0 <= Buy[hamburger] <= +inf
+0 <= Buy[milk] <= +inf
+0 <= Buy[fries] <= +inf
+0 <= Buy[pizza] <= +inf
end
