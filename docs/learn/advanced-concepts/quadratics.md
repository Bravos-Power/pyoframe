# Quadratic expressions

Quadratic expressions work as you'd expect. Simply multiply two linear expression together (or square an expression with `**2`) and you'll get a quadratic. The quadratic can then be used in constraints or the objective.

## Example

### Maximize area of box
Here's a short example that shows that a square maximizes the area of any box with a fixed perimeter.

```python
import pyoframe as pf

model = pf.Model(sense="max")
model.w = pf.Variable(lb=0)
model.h = pf.Variable(lb=0)
model.limit_perimter = 2 * (model.w + model.h) <= 20
model.objective = model.w * model.h
model.optimize()
print(f"It's a square: {model.w.solution == model.h.solution}")
print(f"With area: {model.objective.evaluate()}")

# Outputs:
# It's a square: True
# With area: 25.0
```

<!-- invisible-code-block: python
assert model.w.solution == model.h.solution
assert model.objective.evaluate() == 25
-->

### Facility Location Problem

See [examples/facility_location](../tests/examples/facility_location/).

