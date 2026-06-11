\ Created by MOSEK version 10.2.19
minimize
  + 0 ONE + 2.49 Buy_hamburger + 2.89 Buy_chicken + 1.5 Buy_hot_dog + 1.89 Buy_fries
 + 2.09 Buy_macaroni + 1.99 Buy_pizza + 2.49 Buy_salad + 0.89 Buy_milk + 1.59 Buy_ice_cream
subject to
 Buy_ub_hamburger:  - 40 ONE + Buy_hamburger <= 0
 Buy_ub_chicken:  - 30 ONE + Buy_chicken <= 0
 Buy_ub_hot_dog:  - 20 ONE + Buy_hot_dog <= 0
 Buy_ub_fries:  - 20 ONE + Buy_fries <= 0
 Buy_ub_macaroni:  - 40 ONE + Buy_macaroni <= 0
 Buy_ub_pizza:  - 50 ONE + Buy_pizza <= 0
 Buy_ub_salad:  - 60 ONE + Buy_salad <= 0
 Buy_ub_milk:  - 6.8 ONE + Buy_milk <= 0
 Buy_ub_ice_cream:  - 80 ONE + Buy_ice_cream <= 0
 min_nutrients_calories:  + 1800 ONE - 410 Buy_hamburger - 420 Buy_chicken - 560 Buy_hot_dog
 - 380 Buy_fries - 320 Buy_macaroni - 320 Buy_pizza - 320 Buy_salad - 100 Buy_milk
 - 330 Buy_ice_cream <= 0
 min_nutrients_protein:  + 91 ONE - 24 Buy_hamburger - 32 Buy_chicken - 20 Buy_hot_dog
 - 4 Buy_fries - 12 Buy_macaroni - 15 Buy_pizza - 31 Buy_salad - 8 Buy_milk - 8 Buy_ice_cream <= 0
 max_nutrients_calories:  - 2200 ONE + 410 Buy_hamburger + 420 Buy_chicken + 560 Buy_hot_dog
 + 380 Buy_fries + 320 Buy_macaroni + 320 Buy_pizza + 320 Buy_salad + 100 Buy_milk
 + 330 Buy_ice_cream <= 0
 max_nutrients_fat:  - 65 ONE + 26 Buy_hamburger + 10 Buy_chicken + 32 Buy_hot_dog
 + 19 Buy_fries + 10 Buy_macaroni + 12 Buy_pizza + 12 Buy_salad + 2.5 Buy_milk + 10 Buy_ice_cream <= 0
 max_nutrients_sodium:  - 1779 ONE + 730 Buy_hamburger + 1190 Buy_chicken + 1800 Buy_hot_dog
 + 270 Buy_fries + 930 Buy_macaroni + 820 Buy_pizza + 1230 Buy_salad + 125 Buy_milk
 + 180 Buy_ice_cream <= 0
bounds
 1 <= ONE <= 1
 0 <= Buy_hamburger <= +infinity
 0 <= Buy_chicken <= +infinity
 0 <= Buy_hot_dog <= +infinity
 0 <= Buy_fries <= +infinity
 0 <= Buy_macaroni <= +infinity
 0 <= Buy_pizza <= +infinity
 0 <= Buy_salad <= +infinity
 0 <= Buy_milk <= +infinity
 0 <= Buy_ice_cream <= +infinity
end
