min

obj:

+2.49 BuyFood_hamburger
+2.89 BuyFood_chicken
+1.5 BuyFood_hot_dog
+1.89 BuyFood_fries
+2.09 BuyFood_macaroni
+1.99 BuyFood_pizza
+2.49 BuyFood_salad
+0.89 BuyFood_milk
+1.59 BuyFood_ice_cream


s.t.

MinNutrients_calories:
-410.0 BuyFood_hamburger
-420.0 BuyFood_chicken
-560.0 BuyFood_hot_dog
-380.0 BuyFood_fries
-320.0 BuyFood_macaroni
-320.0 BuyFood_pizza
-320.0 BuyFood_salad
-100.0 BuyFood_milk
-330.0 BuyFood_ice_cream
 <= -1800.0

MinNutrients_fat:
-26.0 BuyFood_hamburger
-10.0 BuyFood_chicken
-32.0 BuyFood_hot_dog
-19.0 BuyFood_fries
-10.0 BuyFood_macaroni
-12.0 BuyFood_pizza
-12.0 BuyFood_salad
-2.5 BuyFood_milk
-10.0 BuyFood_ice_cream
 <= -0.0

MinNutrients_protein:
-24.0 BuyFood_hamburger
-32.0 BuyFood_chicken
-20.0 BuyFood_hot_dog
-4.0 BuyFood_fries
-12.0 BuyFood_macaroni
-15.0 BuyFood_pizza
-31.0 BuyFood_salad
-8.0 BuyFood_milk
-8.0 BuyFood_ice_cream
 <= -91.0

MinNutrients_sodium:
-730.0 BuyFood_hamburger
-1190.0 BuyFood_chicken
-1800.0 BuyFood_hot_dog
-270.0 BuyFood_fries
-930.0 BuyFood_macaroni
-820.0 BuyFood_pizza
-1230.0 BuyFood_salad
-125.0 BuyFood_milk
-180.0 BuyFood_ice_cream
 <= -0.0

MaxNutrients_calories:
+410.0 BuyFood_hamburger
+420.0 BuyFood_chicken
+560.0 BuyFood_hot_dog
+380.0 BuyFood_fries
+320.0 BuyFood_macaroni
+320.0 BuyFood_pizza
+320.0 BuyFood_salad
+100.0 BuyFood_milk
+330.0 BuyFood_ice_cream
 <= 2200.0

MaxNutrients_fat:
+26.0 BuyFood_hamburger
+10.0 BuyFood_chicken
+32.0 BuyFood_hot_dog
+19.0 BuyFood_fries
+10.0 BuyFood_macaroni
+12.0 BuyFood_pizza
+12.0 BuyFood_salad
+2.5 BuyFood_milk
+10.0 BuyFood_ice_cream
 <= 65.0

MaxNutrients_protein:
+24.0 BuyFood_hamburger
+32.0 BuyFood_chicken
+20.0 BuyFood_hot_dog
+4.0 BuyFood_fries
+12.0 BuyFood_macaroni
+15.0 BuyFood_pizza
+31.0 BuyFood_salad
+8.0 BuyFood_milk
+8.0 BuyFood_ice_cream
 <= inf

MaxNutrients_sodium:
+730.0 BuyFood_hamburger
+1190.0 BuyFood_chicken
+1800.0 BuyFood_hot_dog
+270.0 BuyFood_fries
+930.0 BuyFood_macaroni
+820.0 BuyFood_pizza
+1230.0 BuyFood_salad
+125.0 BuyFood_milk
+180.0 BuyFood_ice_cream
 <= 1779.0



bounds

+0 <= BuyFood_hamburger <= inf
+0 <= BuyFood_chicken <= inf
+0 <= BuyFood_hot_dog <= inf
+0 <= BuyFood_fries <= inf
+0 <= BuyFood_macaroni <= inf
+0 <= BuyFood_pizza <= inf
+0 <= BuyFood_salad <= inf
+0 <= BuyFood_milk <= inf
+0 <= BuyFood_ice_cream <= inf
end
