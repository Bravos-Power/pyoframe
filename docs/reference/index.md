# Overview of Pyoframe's API

Pyoframe's classes and subclasses are structured as follows:

- [Model][pyoframe.Model]
- [BaseBlock][pyoframe._model_element.BaseBlock]*
    - [Constraint][pyoframe.Constraint]
    - [BaseOperableBlock][pyoframe._core.BaseOperableBlock]*
        - [Variable][pyoframe.Variable]
        - [Expression][pyoframe.Expression]
            - [Objective][pyoframe.Objective]
        - [Set][pyoframe.Set]
- [Config][pyoframe._Config]

The following enums are also available:

- [VType][pyoframe.VType]
- [ObjSense][pyoframe.ObjSense]

Finally, [PyoframeError][pyoframe.PyoframeError] is a Pyoframe custom error type. 

All the above classes (except those marked with an asterisk) can be imported via:

```python
import pyoframe as pf
```

Additionally, importing Pyoframe patches Pandas and Polars such that the following methods are available.

- [polars.DataFrame.to_expr](./external/polars.DataFrame.to_expr.md)
- [pandas.DataFrame.to_expr](./external/pandas.DataFrame.to_expr.md)
- [pandas.Series.to_expr](./external/pandas.Series.to_expr.md)