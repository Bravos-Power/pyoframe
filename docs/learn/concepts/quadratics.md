# Quadratic expressions

Quadratic expressions work as you'd expect. Simply multiply two linear expression together (or square an expression with `**2`) and you'll get a quadratic. The quadratic can then be used in constraints or the objective.

## Example 1: Maximize area of box
Here's a short example that shows that a square maximizes the area of any box with a fixed perimeter.

```python
import pyoframe as pf

model = pf.Model()
model.w = pf.Variable(lb=0)
model.h = pf.Variable(lb=0)
model.limit_perimter = 2 * (model.w + model.h) <= 20
model.maximize = model.w * model.h
model.optimize()
print(f"It's a square: {model.w.solution == model.h.solution}")
print(f"With area: {model.objective.value}")

# Outputs:
# It's a square: True
# With area: 25.0
```

<!-- invisible-code-block: python
assert model.w.solution == model.h.solution
assert model.objective.value == 25
-->

## Example 2: Facility location problem

See [examples/facility_location](../../examples/facility_location.md).

