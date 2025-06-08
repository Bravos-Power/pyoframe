# Performance

Pyoframe is already one of the fastest and lowest-memory-footprint libraries to build optimization models since we leverage `Polars` internally. Here are some additional tips to squeeze out every bit of performance:

1. **Use [polars](https://pola.rs/) not pandas**. Internally, pyoframe uses polars for everything. If you're using Pandas we'll just convert your dataframes to Polars. So might as well use polars from the very beginning! You'll save time during your pre-processing and data loading.

2. **Use integers not strings for indexing**. Pyoframe works fine with dataframes that contain string columns but you should know that strings take up a lot more space than just numbering your values. When possible, use integer indices.

3. **Tweak the `pf.Config` settings.** Take a look at our [API Reference][pyoframe.Config] and you might find some settings to adjust to squeeze out the last bit of performance.

## `Expression` or `Variable` ?

One common question when building large models is, if you have a very long linear expression, should you assign it to a variable or simply use the expression directly? In some cases, it is best to assign it to a variable since Pyoframe will then only need to pass around the variable rather than all the terms in the linear expression. If you're concerned that you'll be adding more variables to your model, know that most solvers will rapidly and easily get rid of these variables during the presolve stage without any noticeable performance cost.