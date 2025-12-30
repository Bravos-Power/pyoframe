# Addition and its quirks

In Pyoframe, [`Expression`][pyoframe.Expression] objects can be added using the `+` operator, as you might expect.

However, sometimes an addition is ambiguous or indicative of a potential mistake in your model. In these situations, Pyoframe forces you to use _addition modifiers_ to specify exactly how you'd like the addition to be performed. This safety feature helps prevent and quickly fix mistakes in your model.

There are three common addition modifiers in Pyoframe: [`.over(…)`][pyoframe.Expression.over], [`.keep_extras()`][pyoframe.Expression.keep_extras], and [`.drop_extras()`][pyoframe.Expression.drop_extras].

Before delving into these addition modifiers, please note that **these addition rules also apply to subtraction as well as the `<=` and `>=` operators used to create constraints**. This is because subtraction is actually computed as an addition (`a - b` is computed as `a + (-b)`). Similarly, creating a constraint with the `<=` or `>=` operators involves combining the left and right hand sides using addition (`a <= b` becomes `a + (-b) <= 0`). So, although I may only mention addition from now on, please remember that this page also applies to subtraction and to constraint creation.

The rest of the page is organized as follows:

1. [The `.over(…)` addition modifier](#adding-expressions-with-differing-dimensions-using-over)

2. [The `.keep_extras()` and `.drop_extras()` addition modifiers](#handling-extra-labels-with-keep_extras-and-drop_extras)

3. [Important note on the order of operations of addition modifiers](#order-of-operations-for-addition-modifiers)

## Adding expressions with differing dimensions using `.over(…)`

To help catch mistakes, adding expressions with differing dimensions is disallowed by default. [`.over(…)`][pyoframe.Expression.over] overrides this default and **indicates that an addition should be performed by "broadcasting" the differing dimensions.**

The following examples help illustrate when `.over(…)` should and shouldn't be used.

### Example 1: Catching a mistake

Say you're developing an optimization model to study aviation emissions. You'd like to express the sum of in-flight emissions and on-the-ground ([taxiing](https://en.wikipedia.org/wiki/Taxiing)) emissions, _for each flight_, but when you try to add both `Expression` objects, you get an error:

<!-- invisible-code-block: python
import pyoframe as pf
import polars as pl

air_data = pl.DataFrame({"flight_no": ["A4543", "K937"], "emissions": [1.4, 2.4]})

model = pf.Model()
model.Fly = pf.Variable(air_data["flight_no"], vtype="binary")
model.air_emissions = model.Fly * air_data
model.ground_emissions = pf.Param(
    {"flight_number": ["A4543", "K937"], "emissions": [0.02, 0.05]}
)
-->

```pycon
>>> model.flight_emissions = model.air_emissions + model.ground_emissions
Traceback (most recent call last):
...
pyoframe._constants.PyoframeError: Cannot add the two expressions below because their
  dimensions are different (['flight_no'] != ['flight_number']).
Expression 1:  air_emissions
Expression 2:  ground_emissions
If this is intentional, use .over(…) to broadcast. Learn more at
  https://bravos-power.github.io/pyoframe/latest/learn/concepts/addition/#adding-expressions-with-differing-dimensions-using-over

```

Do you understand what happened? The error informs us that `model.air_emissions` has dimension _`flight_no`_, but `model.ground_emissions` has dimension _`flight_number`_. Oops, the two datasets use slightly different spellings! You can use [`.rename(…)`][pyoframe.Expression.rename] to correct for the `Expression` objects having differing dimension names.

```pycon
>>> model.flight_emissions = model.air_emissions + model.ground_emissions.rename({"flight_number": "flight_no"})

```

Benign mistakes like these are relatively common and Pyoframe's error messages help you detect them early. Now, let's examine a case where `.over(…)` is needed.

### Example 2: Broadcasting with `.over(…)`

Say, you'd like to see what happens if, instead of minimizing total emissions, you were to minimize the emissions of the _most emitting flight_. Mathematically, this is equivalent to minimizing variable `E_max` where `E_max` is constrained to be greater or equal to the emissions of every flight.

You try to express this constraint in Pyoframe, but get an error:

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

The error indicates that `E_max` has no dimensions but `flight_emissions` has dimensions `flight_no`. The error is raised because, by default, combining terms with differing dimensions is not allowed (as explained in [example 1](#example-1-catching-a-mistake)).

What we'd like to do is effectively 'copy' (aka. 'broadcast') `E_max` _over_ every flight number. `E_max.over("flight_no")` does just this:

```pycon
>>> model.E_max.over("flight_no")
<Expression (linear) terms=1>
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
<Constraint 'emission_constraint' (linear) height=2 terms=6>
┌───────────┬───────────────────────────────┐
│ flight_no ┆ constraint                    │
│ (2)       ┆                               │
╞═══════════╪═══════════════════════════════╡
│ A4543     ┆ E_max -1.4 Fly[A4543] >= 0.02 │
│ K937      ┆ E_max -2.4 Fly[K937] >= 0.05  │
└───────────┴───────────────────────────────┘

```

## Handling 'extra' labels with `.keep_extras()` and `.drop_extras()`

Addition is performed by pairing the labels in the left `Expression` with those in the right `Expression`. But, what happens when the left and right labels differ?

If one of the two expressions in an addition has extras labels not present in the other, [`.keep_extras()`][pyoframe.Expression.keep_extras] or [`.drop_extras()`][pyoframe.Expression.drop_extras] must be used to indicate how the extra labels should be handled.

### Example 3: Deciding how to handle extra labels

<!-- invisible-code-block: python
import pyoframe as pf
import polars as pl

model = pf.Model()
model.air_emissions = pf.Param(
    {
        "flight_no": ["A4543", "K937", "D2082", "D8432", "D1206"],
        "emissions": [1.4, 2.4, 4, 7.6, 4],
    }
)
model.ground_emissions = pf.Param(
    {"flight_no": ["A4543", "K937", "B3420"], "emissions": [0.02, 0.05, 0.001]}
)
-->

Consider again [example 1](#example-1-catching-a-mistake) where we added air emissions and ground emissions.
Let's say that you fixed the error in example 1, but when you try the addition again you get the following error:

```pycon
>>> model.air_emissions + model.ground_emissions
Traceback (most recent call last):
...
pyoframe._constants.PyoframeError: Cannot add the two expressions below because expression 1 has extra labels.
Expression 1:       air_emissions
Expression 2:       ground_emissions
Extra labels in expression 1:
┌───────────┐
│ flight_no │
╞═══════════╡
│ D2082     │
│ D8432     │
│ D1206     │
└───────────┘
Use .drop_extras() or .keep_extras() to indicate how the extra labels should be handled. Learn more at
    https://bravos-power.github.io/pyoframe/latest/learn/concepts/addition

```

Do you understand what happened? The error explains that `air_emissions` contains flight numbers that are not present in `ground_emissions` (specifically flight numbers `D2082`, `D8432`, and `D1206`). Your ground emissions dataset is missing some flights!

Pyoframe raised an error because it is unclear how you'd like the addition to be performed. In fact, you have three options:

1. Decide to discard all flights with missing ground data (`model.air_emissions.drop_extras()`).
2. Decide to keep all flights, assuming `0` ground emissions when the data is missing (`model.air_emissions.keep_extras()`).
3. Go back to your data processing and fix the root cause of the missing data.

After investigating, you realize that the data is missing because the ground emissions for those flights were negligible. As such, you decide to use `.keep_extras()` (option 2), effectively setting ground emissions to `0` whenever the data is missing.

Let's try again!

```pycon
>>> model.air_emissions.keep_extras() + model.ground_emissions
Traceback (most recent call last):
...
pyoframe._constants.PyoframeError: Cannot add the two expressions below because expression 2 has extra labels.
Expression 1:	air_emissions.keep_extras()
Expression 2:	ground_emissions
Extra labels in expression 2:
┌───────────┐
│ flight_no │
╞═══════════╡
│ B3420     │
└───────────┘
Use .drop_extras() or .keep_extras() to indicate how the extra labels should be handled. Learn more at
    https://bravos-power.github.io/pyoframe/latest/learn/concepts/addition

```

Argh, another error! Do you understand what happened? This time `ground_emissions` has extra labels: flight `B3420` is present in `ground_emissions` but not `air_emissions`. Again, Pyoframe raised an error because it is unclear what should be done:

1. Discard flight `B3420` because the air emissions data is missing.
2. Keep flight `B3420`, assuming the air emissions data is `0`.
3. Go back to your data processing and fix the root cause of the missing air emissions data.

Option 2 hardly seems reasonable this time considering that air emissions make up the majority of a flight's emissions. You end up deciding to discard the flight entirely (option 1) using `.drop_extras()`. Now, the addition works perfectly!

```pycon
>>> model.air_emissions.keep_extras() + model.ground_emissions.drop_extras()
<Expression (parameter) height=5 terms=5>
┌───────────┬────────────┐
│ flight_no ┆ expression │
│ (5)       ┆            │
╞═══════════╪════════════╡
│ A4543     ┆ 1.42       │
│ K937      ┆ 2.45       │
│ D2082     ┆ 4          │
│ D8432     ┆ 7.6        │
│ D1206     ┆ 4          │
└───────────┴────────────┘

```

## Order of operations for addition modifiers

When an operation creates a new [Expression][pyoframe.Expression], any previously applied addition modifiers are discarded to prevent unexpected behaviors. As such, **addition modifiers only work if they're applied _right before_ an addition**. For example, `a.drop_extras().sum("time") + b` won't work but `a.sum("time").drop_extras() + b` will.

There are two exceptions to this rule:

1. _Negation_. Negation preserves addition modifiers. If it weren't for this exception, `-my_obj.drop_extras()` wouldn't work as expected; you would have to write `(-my_obj).drop_extras()` which is unintuitive!

2. _Addition/subtraction_. A `.keep_extras()` or `.drop_extras()` in the left and/or right side of an addition or subtraction is preserved in the result because this allows you to write
    ```
    a.keep_extras() + b.keep_extras() + c.keep_extras()
    ```
    instead of the annoyingly verbose,
    ```
    (a.keep_extras() + b.keep_extras()).keep_extras() + c.keep_extras()
    ```
    (If the left and right sides have conflicting addition modifiers, e.g., `a.keep_extras() + b.drop_extras()`, no addition modifiers are preserved. Also, if you'd like an addition or subtraction _not_ to preserve addition modifiers, you can force the result back to the default of raising errors whenever there are extra labels by using [`.raise_extras()`][pyoframe.Expression.raise_extras].)