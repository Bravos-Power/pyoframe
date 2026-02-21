from amplpy import AMPL
from benchmark_utils.ampl import Benchmark


class Bench(Benchmark):
    def build(self, **kwargs):
        model = AMPL()

        model.eval("""
        param G;
        param F;
        param M := 2*sqrt(2);

        set Grid := 0..G;
        set Facs := 1..F;
        set Dims := 1..2;

        var facility_loc{f in Facs, d in Dims} >= 0, <= 1;
        var max_dist >= 0;
        var is_closest{i in Grid, j in Grid, f in Facs} binary;
        var dist{i in Grid, j in Grid, f in Facs} >= 0;
        var r{i in Grid, j in Grid, f in Facs, d in Dims};

        minimize obj: max_dist;

        # Each grid point is assigned to exactly one facility
        s.t. assignment{i in Grid, j in Grid}:
            sum {f in Facs} is_closest[i,j,f] = 1;

        # Define r (coordinate differences)
        s.t. r_def_x{i in Grid, j in Grid, f in Facs}:
            r[i,j,f,1] = i/G - facility_loc[f,1];

        s.t. r_def_y{i in Grid, j in Grid, f in Facs}:
            r[i,j,f,2] = j/G - facility_loc[f,2];

        # Second-order cone distance
        s.t. soc_dist{i in Grid, j in Grid, f in Facs}:
            r[i,j,f,1]^2 + r[i,j,f,2]^2  <= dist[i,j,f]^2;

        # Big-M linking for max distance
        s.t. bigM_link{i in Grid, j in Grid, f in Facs}:
            dist[i,j,f] == max_dist + M*(1 - is_closest[i,j,f]);
        """)

        model.param["G"] = self.size
        model.param["F"] = self.size

        return model


if __name__ == "__main__":
    bench = Bench(size=4)
    bench.run()
    print(bench.get_objective())
