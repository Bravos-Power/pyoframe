# Facility Location Problem

This is the exact same problem as described in the original [JuMP paper](https://epubs.siam.org/doi/10.1137/15M1020575) (see Section 8.2). It is re-explained here for clarity.

### Problem description

The purpose of this problem is to choose the location of $F$ facilities such that the maximum distance between any customer and its nearest facility is minimized. The $C^2$ Customers are evenly spread out over a square grid (see image below). Facilities can be placed anywhere in the grid.


### Problem formulation

To simplify the formulation, let us scale our square grid such that its axes span go from $0$ to $1$.

We define a variable $y_{f,ax}$ which is the location of every facility ($f$) for each axis ($ax$) (either x-axis or y-axis).

$$ 0 \leq y_{f,ax} \leq 1$$

Next, we define the distance between every facility-customer pair, $s_{i,j,f}$.
 $$r^x_{i,j,f} = (i / N - y_{f,1}) \qquad \forall i, j, f$$
$$r^y_{i,j,f} = (j / N - y_{f,2}) \qquad \forall i,j,f$$

$$s_{i,j,f} ^2 = (r_{i,j,f}^x) ^ 2 + (r^y_{i,j,f}) ^2 \qquad \forall i,j,f$$
$$ s_{i,j,f} \geq 0 \qquad \forall i,j,f$$

We'd like to minimize the maximum distance between any facility and its nearest neighbor $d$.

$$\text{min} \quad d$$

At first, we might try to simply define $d$ as follows:

$$ d^{max} \geq s_{i,j,f} \qquad \forall i,j,f$$

However this is not quite right. We only wish to ensure the maximum distance is bounded by the distances between every customer and its _nearest_ facility. To do this, suppose we had a binary variable, $z_{i,j,f}$ that equals $1$ for customer-facility pairs that are relevant (when the facility is the nearest to the customer) and $0$ otherwise. We can now rewrite the above constraint as follows to fix our issue.

$$ d^{max} \geq s_{i,j,f} - \sqrt{2} (1 - z_{i,j,f}) \qquad \forall i,j,f $$

Why does this work? When $z_{i,j,f} = 0$ the right hand side is necessarily zero or less (since $\sqrt{2}$ is the largest possible distance in a 1-by-1 square) and the constraint is thus non-binding as desired.

Finally, how do we define this magical $z_{i,j,f}$ variable? We simply ensure that exactly one customer-facility pair exists for each customer. The optimization will automatically choose the nearest facility if it matters since it wishes to minimize the objective.

$$\sum_{f} z_{i,j,f} = 1 \qquad \forall i,j$$

