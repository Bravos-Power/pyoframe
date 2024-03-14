minimize

obj:

+0.89 Buy[milk] +2.89 Buy[chicken] +1.5 Buy[hot_dog] +2.09 Buy[macaroni] +1.89 Buy[fries] +1.99 Buy[pizza] +2.49 Buy[salad] +2.49 Buy[hamburger] +1.59 Buy[ice_cream] 

s.t.

con_min_nutrients[calories]: -100.0 Buy[milk] -420.0 Buy[chicken] -560.0 Buy[hot_dog] -320.0 Buy[macaroni] -380.0 Buy[fries] -320.0 Buy[pizza] -320.0 Buy[salad] -410.0 Buy[hamburger] -330.0 Buy[ice_cream]  <= -1800.0
con_min_nutrients[protein]: -8.0 Buy[milk] -32.0 Buy[chicken] -20.0 Buy[hot_dog] -12.0 Buy[macaroni] -4.0 Buy[fries] -15.0 Buy[pizza] -31.0 Buy[salad] -24.0 Buy[hamburger] -8.0 Buy[ice_cream]  <= -91.0
con_max_nutrients[calories]: +100.0 Buy[milk] +420.0 Buy[chicken] +560.0 Buy[hot_dog] +320.0 Buy[macaroni] +380.0 Buy[fries] +320.0 Buy[pizza] +320.0 Buy[salad] +410.0 Buy[hamburger] +330.0 Buy[ice_cream]  <= 2200.0
con_max_nutrients[fat]: +2.5 Buy[milk] +10.0 Buy[chicken] +32.0 Buy[hot_dog] +10.0 Buy[macaroni] +19.0 Buy[fries] +12.0 Buy[pizza] +12.0 Buy[salad] +26.0 Buy[hamburger] +10.0 Buy[ice_cream]  <= 65.0
con_max_nutrients[sodium]: +125.0 Buy[milk] +1190.0 Buy[chicken] +1800.0 Buy[hot_dog] +930.0 Buy[macaroni] +270.0 Buy[fries] +820.0 Buy[pizza] +1230.0 Buy[salad] +730.0 Buy[hamburger] +180.0 Buy[ice_cream]  <= 1779.0


bounds

+0 <= Buy[milk] <= +inf
+0 <= Buy[chicken] <= +inf
+0 <= Buy[hot_dog] <= +inf
+0 <= Buy[macaroni] <= +inf
+0 <= Buy[fries] <= +inf
+0 <= Buy[pizza] <= +inf
+0 <= Buy[salad] <= +inf
+0 <= Buy[hamburger] <= +inf
+0 <= Buy[ice_cream] <= +inf
end
