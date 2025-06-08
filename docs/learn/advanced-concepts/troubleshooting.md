# Troubleshooting

Here are some potential errors and how to resolve them.

### `datatypes of join keys don't match`

Often, this error indicates that two columns in your input dataframes, although representing the same dimension, have different datatypes (e.g. 16bit integer and 64bit integer). This is not allowed and you should ensure that for the same dimensions, datatypes are identical.