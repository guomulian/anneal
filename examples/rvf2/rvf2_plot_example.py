from anneal import anneal
from examples.rvf2 import rvf2
import random

import os
import copy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


if __name__ == '__main__':
    DIR_NAME = os.path.dirname(os.path.abspath(__file__))
    PICKLE_NAME = 'rvf2.pickle'
    FILENAME = os.path.join(DIR_NAME, PICKLE_NAME)

    def fun_1(x, y):
        return x**4 - 3*x**2 + y**4 - 3*y**2 + 1

    def fun_2(x, y):
        return x**3 + y**3

    def fun_3(x, y):
        return np.sin(x) + y**2

    def fun_4(x, y):
        return x**2 + y**2

    def fun_5(x, y):
        return np.sin(x*y)

    # random.seed(0)
    fun = fun_3
    initial_point = (1, 0)
    bounds = [[-2, 2], [-2, 2]]
    max_steps = 100
    objective = 'min'
    solver = rvf2.Rvf2(fun, initial_point, max_steps, bounds, objective)

    best_state, best_energy = solver.anneal(
                                verbose=2,
                                debug=True,
                                filename=FILENAME
                                )

    intermediate_states = solver.unpickle_states(FILENAME)

    xp = np.array([s[0] for s in intermediate_states])
    yp = np.array([s[1] for s in intermediate_states])
    zp = fun(xp, yp)
    sp = np.array(range(len(intermediate_states)))
    ep = np.array(list(map(solver._energy, intermediate_states)))
    tp = np.array(list(map(solver.temperature, sp)))

    steps = 100
    x = np.linspace(*bounds[0], num=steps)
    y = np.linspace(*bounds[1], num=steps)

    X, Y = np.meshgrid(x, y)
    Z = fun(X, Y)

    # Begin Plotly
    def title_string(max_steps, seed=None):
        t = "<b>anneal()</b> results for <b>max_steps</b> = {}"
        if seed:
            t += " with <b>seed</b> = {}"
            t = t.format(max_steps, seed)
        else:
            t = t.format(max_steps)

        return t

    def hover_string(step, state, energy, temp):
        s = "<b>Step:</b> {}"
        s += "<br><b>State:</b> {}"
        s += "<br><b>Energy:</b> {}"
        s += "<br><b>Temperature:</b> {}"

        return s.format(step, state, energy, temp)

    hovertext = [hover_string(sp[i], (xp[i], yp[i]), ep[i], tp[i])
                 for i in range(len(sp))]

    fig = go.Figure()

    fig.update_layout(title=go.layout.Title(text=title_string(max_steps),
                                            xref="paper", x=0))

    fig.add_trace(go.Scatter(x=sp,
                             y=ep,
                             mode='lines+markers',
                             marker=dict(size=5,
                                         line=dict(width=1,
                                                   color='darkslategray')),
                             showlegend=False,
                             hoverinfo="text",
                             hovertext=hovertext))

    fig.show()

    print("Solution: {}\nMin Value: {}\n".format(best_state, best_energy))
