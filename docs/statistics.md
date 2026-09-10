# Statistics nodes

`Median`, `Mode`, `Variance`, `StandardDeviation`, `Range`, `Percentile`, and
`Quantile` take `values: SequenceValue[FloatValue]` and return `value: FloatValue`.
Integer sequences can feed these inputs through the existing numeric casts.
All nodes are version `1.0.0` and are registered as package entry points.

| Node | Behavior | Parameters |
| --- | --- | --- |
| `Median` | Middle sorted value; mean of the middle pair for even length | None |
| `Mode` | Most frequent value; first encountered value wins a tie | None |
| `Variance` | Sum of squared deviations divided by `n-1` for a sample or `n` for a population | `population=false` |
| `StandardDeviation` | Square root of the selected variance | `population=false` |
| `Range` | Largest value minus smallest value | None |
| `Percentile` | Quantile requested on the 0–100 scale, including endpoints | Required `q`; `interpolation="linear"` |
| `Quantile` | Quantile requested on the 0–1 scale, including endpoints | Required `q`; `interpolation="linear"` |

An empty numeric sequence raises a user-visible validation error. Sample variance
and standard deviation also require at least two values. Population versions
accept a singleton and return zero. A mode exists for any nonempty sequence:
with no repeated values, it is the first input value. These choices are stable
contracts, including their behavior when used inside an `Attempt` boundary.

Calculations use the existing Decimal-backed `FloatValue` directly. Median,
mode, variance, and standard deviation use Python's standard-library statistics
operations over Decimal values; range and interpolation also stay in Decimal.
There is no intermediate binary-float conversion. Decimal precision and rounding
follow the active Decimal context; the final JSON numeric representation follows
the engine's existing FloatValue serialization contract.

## Quantile interpolation

Sort the `n` values and compute position `h = (n-1) * q`, where percentile `q`
is first divided by 100. The endpoints return the minimum and maximum. A
singleton returns its only value at every valid quantile. At an exact integer
position, every method returns that position's value. Rank and index selection
are exact even under a low-precision Decimal context; arithmetic on the selected
values still follows the active context's precision and rounding.

Between positions `i = floor(h)` and `i+1`:

| `interpolation` | Result |
| --- | --- |
| `linear` | `values[i] + (values[i+1] - values[i]) * (h-i)` |
| `lower` | `values[i]` |
| `higher` | `values[i+1]` |
| `midpoint` | Mean of the adjacent values |
| `nearest` | Value at the nearest index; exact half-index ties choose the even index |

For `[0, 10, 20, 30]`, the 25th percentile has position `0.75`: linear gives
`7.5`, lower `0`, higher `10`, midpoint `5`, and nearest `10`. Out-of-range `q`
and unknown interpolation methods are rejected when constructing the node.
The interpolation method and population flag have explicit portable defaults
in their parameter schemas, so a serialized graph retains the selected rule.

## Counting with Length

`Length` is the shared generic sequence operation used for counting; there is no
separate numeric `Count` node. It accepts `sequence: SequenceValue[T]`, returns
`length: IntegerValue`, and requires `element_schema` in its params. For example,
`FloatValue.to_value_schema()` declares a numeric sequence and
`StringValue.to_value_schema()` declares text elements. Schemas resolve through
registered value identity, preserving custom value validation.

Length counts every element, including an err-tagged `Result` element. Unlike
an undefined numeric summary, the length of an empty sequence is defined: zero.
It performs no numeric cast, error filtering, or payload inspection.
