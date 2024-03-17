minimize

obj:

2.49 Buy[hamburger] +2.89 Buy[chicken] +1.5 Buy[hot_dog] +1.89 Buy[fries] +2.09 Buy[macaroni] +1.99 Buy[pizza] +2.49 Buy[salad] +0.89 Buy[milk] +1.59 Buy[ice_cream]

s.t.

min_nutrients[calories]: -410.0 Buy[hamburger] -420.0 Buy[chicken] -560.0 Buy[hot_dog] -380.0 Buy[fries] -320.0 Buy[macaroni] -320.0 Buy[pizza] -320.0 Buy[salad] -100.0 Buy[milk] -330.0 Buy[ice_cream] <= -1800.0
min_nutrients[protein]: -24.0 Buy[hamburger] -32.0 Buy[chicken] -20.0 Buy[hot_dog] -4.0 Buy[fries] -12.0 Buy[macaroni] -15.0 Buy[pizza] -31.0 Buy[salad] -8.0 Buy[milk] -8.0 Buy[ice_cream] <= -91.0
max_nutrients[calories]: 410.0 Buy[hamburger] +420.0 Buy[chicken] +560.0 Buy[hot_dog] +380.0 Buy[fries] +320.0 Buy[macaroni] +320.0 Buy[pizza] +320.0 Buy[salad] +100.0 Buy[milk] +330.0 Buy[ice_cream] <= 2200.0
max_nutrients[fat]: 26.0 Buy[hamburger] +10.0 Buy[chicken] +32.0 Buy[hot_dog] +19.0 Buy[fries] +10.0 Buy[macaroni] +12.0 Buy[pizza] +12.0 Buy[salad] +2.5 Buy[milk] +10.0 Buy[ice_cream] <= 65.0
max_nutrients[sodium]: 730.0 Buy[hamburger] +1190.0 Buy[chicken] +1800.0 Buy[hot_dog] +270.0 Buy[fries] +930.0 Buy[macaroni] +820.0 Buy[pizza] +1230.0 Buy[salad] +125.0 Buy[milk] +180.0 Buy[ice_cream] <= 1779.0


bounds

+0 <= Buy[ice_cream] <= +inf
+0 <= Buy[hot_dog] <= +inf
+0 <= Buy[chicken] <= +inf
+0 <= Buy[milk] <= +inf
+0 <= Buy[fries] <= +inf
+0 <= Buy[pizza] <= +inf
+0 <= Buy[salad] <= +inf
+0 <= Buy[macaroni] <= +inf
+0 <= Buy[hamburger] <= +inf
end
