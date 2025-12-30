# Solve a simple problem

<!-- invisible-code-block: python
import os

os.chdir(os.path.join(os.getcwd(), "docs/learn/get-started/basic-example"))
-->

To start, you will solve a simple optimization problem with Pyoframe.

!!! quote "Problem statement"

    Imagine you're a vegetarian hesitating between tofu and chickpeas as
    a source of protein for tomorrow's dinner. You'd like to spend as little money as
    possible while still consuming at least 50g of protein. How many blocks
    of tofu ($4 each, 18g of protein) and cans of chickpeas ($3 each, 15g of protein) should you buy?

Click on the :material-plus-circle: buttons below to understand the code, and then run it on your computer.

```python
import pyoframe as pf

m = pf.Model()

# You can buy tofu or chickpeas
m.tofu_blocks = pf.Variable(lb=0, vtype="integer")  # (1)!
m.chickpea_cans = pf.Variable(lb=0, vtype="integer")

# You want to minimize your cost
m.minimize = 4 * m.tofu_blocks + 3 * m.chickpea_cans  # (2)!

# But still consume enough protein
m.protein_constraint = 18 * m.tofu_blocks + 15 * m.chickpea_cans >= 50  # (3)!

m.optimize()  # (4)!

print("Tofu blocks:", m.tofu_blocks.solution)
print("Chickpea cans:", m.chickpea_cans.solution)
```

<!-- invisible-code-block: python
assert m.tofu_blocks.solution == 2
assert m.chickpea_cans.solution == 1
-->

1. `lb=0` set a lower bound so that you can't buy negative amounts of tofu. 
    
    `vtype="integer"` ensures that you can't buy a fraction of a block.

2. `minimize` and `maximize` are reserved variable names that can be used to set the objective.
3. Pyoframe constraints are easily created with the `<=`, `>=`, or `==` operators.
4. Pyoframe automatically detects the solver you have installed; no need to specify it!

After running the code you should get:

```console
Tofu blocks: 2
Chickpea cans: 1
```

On the [next page](./example-with-dimensions.md), you'll resolve this problem but instead of hard-coded values, you'll use DataFrames, Pyoframe's secret sauce!