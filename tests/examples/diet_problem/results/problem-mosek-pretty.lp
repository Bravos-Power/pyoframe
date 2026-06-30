\ Created by MOSEK version 11.0.27
minimize
  + 0 ONE + 2.49 Buy_veggieburger + 2.89 Buy_tofu + 1.5 Buy_lentils + 1.89 Buy_chickpeas
 + 2.09 Buy_macaroni + 1.99 Buy_pizza + 2.49 Buy_salad + 0.89 Buy_milk + 1.59 Buy_ice_cream
subject to
 Buy_ub_veggieburger:  - 40 ONE + Buy_veggieburger <= 0
 Buy_ub_tofu:  - 30 ONE + Buy_tofu <= 0
 Buy_ub_lentils:  - 20 ONE + Buy_lentils <= 0
 Buy_ub_chickpeas:  - 20 ONE + Buy_chickpeas <= 0
 Buy_ub_macaroni:  - 40 ONE + Buy_macaroni <= 0
 Buy_ub_pizza:  - 50 ONE + Buy_pizza <= 0
 Buy_ub_salad:  - 60 ONE + Buy_salad <= 0
 Buy_ub_milk:  - 6.8 ONE + Buy_milk <= 0
 Buy_ub_ice_cream:  - 80 ONE + Buy_ice_cream <= 0
 min_nutrients_calories:  + 1800 ONE - 410 Buy_veggieburger - 420 Buy_tofu - 560 Buy_lentils
 - 380 Buy_chickpeas - 320 Buy_macaroni - 320 Buy_pizza - 320 Buy_salad - 100 Buy_milk
 - 330 Buy_ice_cream <= 0
 min_nutrients_protein:  + 91 ONE - 24 Buy_veggieburger - 32 Buy_tofu - 20 Buy_lentils
 - 4 Buy_chickpeas - 12 Buy_macaroni - 15 Buy_pizza - 31 Buy_salad - 8 Buy_milk - 8 Buy_ice_cream <= 0
 max_nutrients_calories:  - 2200 ONE + 410 Buy_veggieburger + 420 Buy_tofu + 560 Buy_lentils
 + 380 Buy_chickpeas + 320 Buy_macaroni + 320 Buy_pizza + 320 Buy_salad + 100 Buy_milk
 + 330 Buy_ice_cream <= 0
 max_nutrients_fat:  - 65 ONE + 26 Buy_veggieburger + 10 Buy_tofu + 32 Buy_lentils
 + 19 Buy_chickpeas + 10 Buy_macaroni + 12 Buy_pizza + 12 Buy_salad + 2.5 Buy_milk
 + 10 Buy_ice_cream <= 0
 max_nutrients_sodium:  - 1779 ONE + 730 Buy_veggieburger + 1190 Buy_tofu + 1800 Buy_lentils
 + 270 Buy_chickpeas + 930 Buy_macaroni + 820 Buy_pizza + 1230 Buy_salad + 125 Buy_milk
 + 180 Buy_ice_cream <= 0
bounds
 1 <= ONE <= 1
 0 <= Buy_veggieburger <= +infinity
 0 <= Buy_tofu <= +infinity
 0 <= Buy_lentils <= +infinity
 0 <= Buy_chickpeas <= +infinity
 0 <= Buy_macaroni <= +infinity
 0 <= Buy_pizza <= +infinity
 0 <= Buy_salad <= +infinity
 0 <= Buy_milk <= +infinity
 0 <= Buy_ice_cream <= +infinity
end
