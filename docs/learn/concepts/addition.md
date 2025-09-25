# Addition and its quirks

In Pyoframe, [`Expression`][pyoframe.Expression] objects can be added using the `+` operator, as you might expect.

However, sometimes an addition is ambiguous or indicative of a potential mistake in your model.
In these situations, Pyoframe forces you to use _addition modifiers_ to specify exactly how you'd like the addition to be performed.
This safety feature helps prevent and quickly fix mistakes in your model.

There are three addition modifiers in Pyoframe: [`.over(…)`][pyoframe.Expression.over], [`.keep_extras()`][pyoframe.Expression.keep_extras], and [`.drop_extras()`][pyoframe.Expression.drop_extras].

Before delving into these addition modifiers, it is important to note that **these addition rules also apply to subtraction and constraint creation**. Indeed, subtraction in Pyoframe is computed as an addition (`a - b` is computed as `a + (-1 * b)`). Similarly, whenever the `<=` or `>=` operators are used to create a constraint, the constraint's left and right hand side are combined using addition (`a <= b` becomes `a + (-1 * b) <= 0`). So, although I'll only mention addition from now on, please remember that the following sections also apply to subtraction and constraint creation.


!!! warning "Order of operations for addition modifiers"

    Addition modifiers must be applied _after_ all other operations.[^1] For example, `my_obj.drop_extras().sum("time")` won't work but `my_obj.sum("time").drop_extras()` will. (This rule prevents unexpected behaviors where addition modifiers "survive" through multiple operations.)

[^1]: The exception to this rule is negation. As one might expect, `-my_obj.drop_extras()` works the same as `(-my_obj).drop_extras()` even though, in the former case, a negation is applied _after_ the addition modifier.

## Adding expressions with differing dimensions using `.over(…)`

To help catch mistakes, adding expressions with differing dimensions is disallowed by default. [`.over(…)`][pyoframe.Expression.over] overrides this default and **indicates that an addition should be performed by "broadcasting" the differing dimensions.**

The following example helps illustrate when `.over(…)` should and shouldn't be used.

Say you're developing an optimization model to study aviation emissions. You'd like to add the in-flight emissions with the [taxiing](https://en.wikipedia.org/wiki/Taxiing) emissions to create an `Expression` representing the total emissions on a flight-by-flight basis. Unfortunately, doing so gives an error:

<!-- invisible-code-block: python
import pyoframe as pf
import polars as pl

air_data = pl.DataFrame({"flight_no": ["A4543", "K937"], "emissions": [1.4, 2.4]})
ground_data = pl.DataFrame(
    {"flight_number": ["A4543", "K937"], "emissions": [0.02, 0.05]}
)

model = pf.Model()
model.Fly = pf.Variable(air_data["flight_no"], vtype="binary")
model.air_emissions_by_flight = model.Fly * air_data
model.ground_emissions_by_flight = ground_data.to_expr()
-->

```pycon
>>> model.air_emissions_by_flight + model.ground_emissions_by_flight
Traceback (most recent call last):
...
pyoframe._constants.PyoframeError: Cannot add the two expressions below because their
  dimensions are different (['flight_no'] != ['flight_number']).
Expression 1:  air_emissions_by_flight
Expression 2:  ground_emissions_by_flight
If this is intentional, use .over(…) to broadcast. Learn more at
  https://bravos-power.github.io/pyoframe/latest/learn/concepts/addition/#adding-expressions-with-differing-dimensions-using-over

```

This error helps you catch a mistake. The error informs us that `model.air_emissions_by_flight` has dimension _`flight_no`_ but `model.ground_emissions_by_flight` has dimension _`flight_number`_. Oops, they're spelt differently! Seems like the two datasets containing the emissions data had slightly different column names.

Benign mistakes like these are relatively common and Pyoframe's defaults help you catch these mistakes early. Now, let's examine a case where `.over(…)` is needed.

Say, you'd like to see what happens if, instead of minimizing total emissions, you were to minimize the emissions of the _most emitting flight_. Mathematically, you'd like to minimize $`E_{max}`$ where
$`E_{max} \geq e_i`$ for every flight $`i`$ with emissions $`e_i`$.

You might try the following in Pyoframe, but will get an error:

<!-- invisible-code-block: python
model.flight_emissions = (
    model.air_emissions_by_flight
    + model.ground_emissions_by_flight.rename({"flight_number": "flight_no"})
)
-->

```pycon
>>> model.E_max = pf.Variable()
>>> model.minimize = model.E_max
>>> model.emission_constraint = model.E_max >= model.flight_emissions
Traceback (most recent call last):
...
pyoframe._constants.PyoframeError: Cannot subtract the two expressions below because their
    dimensions are different ([] != ['flight_no']).
Expression 1:  E_max
Expression 2:  flight_emissions
If this is intentional, use .over(…) to broadcast. Learn more at
    https://bravos-power.github.io/pyoframe/latest/learn/concepts/addition/#adding-expressions-with-differing-dimensions-using-over

```

The error indicates that `E_max` has no dimensions while `flight_emissions` has dimensions `flight_no`. The error is raised because, by default, combining terms with differing dimensions is not allowed.

What we'd like to do is effectively 'copy' (aka. 'broadcast') `E_max` _over_ every flight number. `E_max.over("flight_no")` does just this:

```pycon
>>> model.E_max.over("flight_no")
<Expression terms=1 type=linear>
┌───────────┬────────────┐
│ flight_no ┆ expression │
╞═══════════╪════════════╡
│ *         ┆ E_max      │
└───────────┴────────────┘

```

Notice how applying `.over("flight_no")` added a dimension `flight_no` with value `*`. The asterix (`*`) indicates that `flight_no` will take the shape of whichever expression `E_max` is later combined with. Since `E_max` is being combined with `flight_emissions`, `*` will be replaced with an entry for every flight number in `flight_emissions`. Now creating our constraint works properly:

```pycon
>>> model.emission_constraint = model.E_max.over("flight_no") >= model.flight_emissions
>>> model.emission_constraint
<Constraint 'emission_constraint' height=2 terms=6 type=linear>
┌───────────┬───────────────────────────────┐
│ flight_no ┆ constraint                    │
│ (2)       ┆                               │
╞═══════════╪═══════════════════════════════╡
│ A4543     ┆ E_max -1.4 Fly[A4543] >= 0.02 │
│ K937      ┆ E_max -2.4 Fly[K937] >= 0.05  │
└───────────┴───────────────────────────────┘

```

## Handling extra values with `.keep_extras()` and `.drop_extras()`

!!! info "Work in progress"

    This documentation could use some help. [Learn how you can contribute](../../contribute/index.md).
