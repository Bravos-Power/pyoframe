# Create constraints

Create constraints by using the `<=`, `>=`, and `==` operators between two expressions. For example,

<!-- invisible-code-block: python
import pyoframe as pf

m = pf.Model()
m.Hours_Worked = pf.Variable({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"]})

-->

```python
m.Con_Max_Weekly_Hours = m.Hours_Worked.sum() <= 40
```

!!! tip "Naming constraints"
    I like prefixing constraint names with `Con_` to easily distinguish them from other module attributes.


## Handle extra labels

When creating constraints, Pyoframe always merges the left- and right-hand side expressions into a single expression (e.g. `a <= b` becomes `(a - b) <= 0`). Thus, if the left- and/or right-hand sides have labels not present in the other side, you will need to handle these extra labels using `drop_extras()` or `keep_extras()`. Read [Addition and its quirks](../concepts/addition.md) to learn more or see the [diet problem](../../examples/diet.md) for an example.

## Relax a constraint

Refer to the API documentation for [`.relax()`][pyoframe.Constraint.relax].