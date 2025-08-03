# Performance tips

Pyoframe is already one of the fastest and most memory-efficient libraries for formulating optimization models. However, if you'd like to squeeze out every bit of performance, here are some additional tips.

## Use polars instead of pandas

[Polars](https://pola.rs/) is much faster than Pandas. Moreover, if you use Pandas, there will be a (very small) overhead because Pyoframe converts all DataFrames to Polars prior to computations.

## Use integer indices

Pyoframe will work with all types of indices, however integer indices are faster and more memory efficient than alternatives (e.g. string indices).

## Disable `maintain_order`

By default, Pyoframe ensures that the order of variables, constraints, and mathematical terms is maintained across runs to ensure that your results are reproducible down to the very last decimal place. However, if you're not bothered by miniscule variations in your results due to numerical errors accumulating differently for different orderings, you should disable [`maintain_order`][pyoframe._Config.maintain_order]:

```python
pf.Config.maintain_order = False
```

## Disable unmatched checks

Disabling unmatched checks means that, instead of raising [unmatched term exceptions](../concepts/special-functions.md#drop_unmatched-and-keep_unmatched), pyoframe will process sums with unmatched terms as if [`keep_unmatched`][pyoframe.Expression.keep_unmatched] had been applied. While this may improve performance, it will silence potentially important errors meant to help you build your model. If you'd like to disable unmatched checks, we recommend you do so only after thoroughly testing your model and ensuring that all potential unmatched term exceptions have been handled.

The following code disables unmatched checks:

```python
pf.Config.disable_unmatched_checks = True
```

<!-- TODO REVISIT

## `Expression` or `Variable` ?

One common question when building large models is, if you have a very long linear expression, should you assign it to a variable or simply use the expression directly? In some cases, it is best to assign it to a variable since Pyoframe will then only need to pass around the variable rather than all the terms in the linear expression. If you're concerned that you'll be adding more variables to your model, know that most solvers will rapidly and easily get rid of these variables during the presolve stage without any noticeable performance cost.
-->
