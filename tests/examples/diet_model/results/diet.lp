minimize

obj:

+2.09 Buy[macaroni] +1.89 Buy[fries] +2.49 Buy[salad] +2.89 Buy[chicken] +1.99 Buy[pizza] +2.49 Buy[hamburger] +1.59 Buy[ice_cream] +0.89 Buy[milk] +1.5 Buy[hot_dog] 

s.t.

min_nutrients[calories]: -320.0 Buy[macaroni] -380.0 Buy[fries] -320.0 Buy[salad] -420.0 Buy[chicken] -320.0 Buy[pizza] -410.0 Buy[hamburger] -330.0 Buy[ice_cream] -100.0 Buy[milk] -560.0 Buy[hot_dog]  <= -1800.0
min_nutrients[protein]: -12.0 Buy[macaroni] -4.0 Buy[fries] -31.0 Buy[salad] -32.0 Buy[chicken] -15.0 Buy[pizza] -24.0 Buy[hamburger] -8.0 Buy[ice_cream] -8.0 Buy[milk] -20.0 Buy[hot_dog]  <= -91.0
max_nutrients[calories]: +320.0 Buy[macaroni] +380.0 Buy[fries] +320.0 Buy[salad] +420.0 Buy[chicken] +320.0 Buy[pizza] +410.0 Buy[hamburger] +330.0 Buy[ice_cream] +100.0 Buy[milk] +560.0 Buy[hot_dog]  <= 2200.0
max_nutrients[fat]: +10.0 Buy[macaroni] +19.0 Buy[fries] +12.0 Buy[salad] +10.0 Buy[chicken] +12.0 Buy[pizza] +26.0 Buy[hamburger] +10.0 Buy[ice_cream] +2.5 Buy[milk] +32.0 Buy[hot_dog]  <= 65.0
max_nutrients[sodium]: +930.0 Buy[macaroni] +270.0 Buy[fries] +1230.0 Buy[salad] +1190.0 Buy[chicken] +820.0 Buy[pizza] +730.0 Buy[hamburger] +180.0 Buy[ice_cream] +125.0 Buy[milk] +1800.0 Buy[hot_dog]  <= 1779.0


bounds

+0 <= Buy[macaroni] <= +inf
+0 <= Buy[fries] <= +inf
+0 <= Buy[salad] <= +inf
+0 <= Buy[chicken] <= +inf
+0 <= Buy[pizza] <= +inf
+0 <= Buy[hamburger] <= +inf
+0 <= Buy[ice_cream] <= +inf
+0 <= Buy[milk] <= +inf
+0 <= Buy[hot_dog] <= +inf
end
