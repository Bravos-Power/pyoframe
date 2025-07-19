# Facility Location Problem

## Problem Statement

This problem, as described in [elsewhere](), seeks to find the optimal `(x,y)` location for a set of `F` facilities such that
the maximum distance between any customer and its nearest facility is minimized. Customers are spread out evenly on a `G`-by-`G` grid.

## Model


```python
import pandas as pd

from pyoframe import Model, Set, sum, Variable

G = 4
F = 3

# Construct model
model = Model()

# Define sets
model.facilities = Set(f=range(F))
model.x_axis = Set(x=range(G))
model.y_axis = Set(y=range(G))
model.axis = Set(d=[1, 2])
model.customers = model.x_axis * model.y_axis


model.facility_position = Variable(model.facilities, model.axis, lb=0, ub=1)
model.customer_position_x = pd.DataFrame(
    {"x": range(G), "x_pos": [step / (G - 1) for step in range(G)]}
).to_expr()
model.customer_position_y = pd.DataFrame(
    {"y": range(G), "y_pos": [step / (G - 1) for step in range(G)]}
).to_expr()

model.max_distance = Variable(lb=0)

model.is_closest = Variable(model.customers, model.facilities, vtype="binary")
model.con_only_one_closest = sum("f", model.is_closest) == 1

model.dist_x = Variable(model.x_axis, model.facilities)
model.dist_y = Variable(model.y_axis, model.facilities)
model.con_dist_x = model.dist_x == model.customer_position_x.add_dim(
    "f"
) - model.facility_position.pick(d=1).add_dim("x")
model.con_dist_y = model.dist_y == model.customer_position_y.add_dim(
    "f"
) - model.facility_position.pick(d=2).add_dim("y")
model.dist = Variable(model.x_axis, model.y_axis, model.facilities, lb=0)
model.con_dist = model.dist**2 == (model.dist_x**2).add_dim("y") + (
    model.dist_y**2
).add_dim("x")

# Twice the max distance which ensures that when is_closest is 0, the constraint is not binding.
M = 2 * 1.414
model.con_max_distance = model.max_distance.add_dim("x", "y", "f") >= model.dist - M * (
    1 - model.is_closest
)

model.minimize = model.max_distance

# Solve model
model.optimize()
```

So what's the maximum distance from a customer to its nearest facility?

```pycon
>>> model.objective.value
0.499999406245417

```