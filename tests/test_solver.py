import pyoframe as pf


def test_retrieving_duals():
    m = pf.Model()

    m.A = pf.Variable()
    m.B = pf.Variable(ub=10)
    m.max_AB = 2 * m.A + m.B <= 100
    m.extra_slack_constraint = 2 * m.A + m.B <= 150
    m.maximize = 0.2 * m.A + 2 * m.B

    m.solve("gurobi")

    assert m.A.solution == 45
    assert m.B.solution == 10
    assert m.objective.value == 29
    assert m.max_AB.dual == 0.1
    assert m.max_AB.slack == 0
    assert m.extra_slack_constraint.dual == 0
    assert m.extra_slack_constraint.slack == 50
    assert m.A.RC == 0
    assert m.B.RC == 1.9
