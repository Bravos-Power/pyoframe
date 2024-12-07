# A simple model

Here's a simple problem to show you Pyoframe's syntax. Click on the :material-plus-circle: buttons to learn what's happening!

> A block of tofu costs $4 and contains 10 g of protein. A can of chickpeas costs $2 and contains 8 g of protein. How should you spend your $10 budget to get the most protein?

Run the code below to find the answer!

```python3
import pyoframe as pf

m = pf.Model("max") # (1)!

m.tofu = pf.Variable(lb=0)  # (2)!
m.chickpeas = pf.Variable(lb=0)

m.objective = 10 * m.tofu + 8 * m.chickpeas # (3)!
m.budget_constraint = 4 * m.tofu + 2 * m.chickpeas <= 10 # (4)!

m.optimize()

print(f"{m.tofu.solution} blocks of tofu")
print(f"{m.chickpeas.solution} cans of chickpeas")
```

1. Creating your model is always the starting point!
2. `lb=0` sets the variable's *lower bound* to ensure you can't buy a negative quantity of tofu!
3. Variables can be added and multiplied as you'd expect!
4. Constraints are easily created with `<=`, `>=` or `==`.

