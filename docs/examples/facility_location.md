# Facility location Problem

## Problem statement

As described in [this paper](https://mlubin.github.io/pdf/jump-sirev.pdf), the facility location problem seeks to find the optimal `(x,y)` location for a set of `F` facilities such that the maximum distance between any customer and its nearest facility is minimized. Customers are spread out evenly on a `G`-by-`G` grid.

## Model

```python
import pandas as pd

import pyoframe as pf

G = 4
F = 3

# Construct model
model = pf.Model()

# Define sets
model.facilities = pf.Set(f=range(F))
model.x_axis = pf.Set(x=range(G))
model.y_axis = pf.Set(y=range(G))
model.axis = pf.Set(axis=["x", "y"])
model.customers = model.x_axis * model.y_axis  # (1)!


model.facility_position = pf.Variable(model.facilities, model.axis, lb=0, ub=1)
model.customer_position_x = pf.Param(
    {"x": range(G), "x_pos": [step / (G - 1) for step in range(G)]}
)
model.customer_position_y = pf.Param(
    {"y": range(G), "y_pos": [step / (G - 1) for step in range(G)]}
)

model.max_distance = pf.Variable(lb=0)

model.is_closest = pf.Variable(model.customers, model.facilities, vtype="binary")
model.con_only_one_closest = model.is_closest.sum("f") == 1

model.dist_x = pf.Variable(model.x_axis, model.facilities)
model.dist_y = pf.Variable(model.y_axis, model.facilities)
model.con_dist_x = model.dist_x == model.customer_position_x.over(
    "f"
) - model.facility_position.pick(axis="x").over("x")
model.con_dist_y = model.dist_y == model.customer_position_y.over(
    "f"
) - model.facility_position.pick(axis="y").over("y")
model.dist = pf.Variable(model.x_axis, model.y_axis, model.facilities, lb=0)
model.con_dist = model.dist**2 == (model.dist_x**2).over("y") + (model.dist_y**2).over(
    "x"
)

M = 2 * 1.414
model.con_max_distance = model.max_distance.over("x", "y", "f") >= model.dist - M * (
    1 - model.is_closest
)

model.minimize = model.max_distance

# Solve model
model.optimize()
```

1. Multiplying [Sets][pyoframe.Set] returns their [cartesian product](https://en.wikipedia.org/wiki/Cartesian_product).

So what's the maximum distance from a customer to its nearest facility?

```pycon
>>> model.objective.value
0.5

```
