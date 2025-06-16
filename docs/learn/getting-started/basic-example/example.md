# Basic example

<!-- invisible-code-block: python
import os

os.chdir(os.path.join(os.getcwd(), "docs/learn/getting-started/basic-example"))
-->

Let's solve the following problem using Pyoframe!

> Imagine you're a vegetarian hesitating between tofu and chickpeas as
> a source of protein for tomorrow's dinner. You'd like to spend as little as
> possible while still consuming at least 50 grams of protein. How many blocks
> of tofu ($4, 18g of protein) and cans of chickpeas ($3, 15g of protein) should you buy?

Click on the :material-plus-circle: buttons below to understand the Pyoframe solution.

```python
import pyoframe as pf

m = pf.Model()

# You can buy tofu or chickpeas
m.tofu_blocks = pf.Variable(lb=0, vtype="integer")  # (1)!
m.chickpea_cans = pf.Variable(lb=0, vtype="integer")

# You want to minimize your cost (4$ per tofu block, $2 per chickpea can)
m.minimize = 4 * m.tofu_blocks + 3 * m.chickpea_cans  # (2)!

# But still consume 10 grams of protein (tofu = 8g, chickpea cans = 6g)
m.protein_constraint = 18 * m.tofu_blocks + 15 * m.chickpea_cans >= 50  # (3)!

m.optimize()  # (4)!
```

1. Notice how we set a lower bound (`lb=0`) so that you can't buy negative amounts. We also specify 
2. `.minimize` and `.maximize` are reserved variables that let you define your objective function.
3. Constraints are easily created with `<=`, `>=`, or `==`.
4. Pyoframe automatically detects your installed solver!

So how much should you buy...

```pycon
>>> m.tofu_blocks.solution
2
>>> m.chickpea_cans.solution
1

```
