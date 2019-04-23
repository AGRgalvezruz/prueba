
# Iris GP

## Main program


```python
import random
import operator
import csv
import itertools

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Read the iris list features and put it in a list of lists.
with open("iris.csv") as irisbase:
    irisReader = csv.reader(irisbase)
    iris = list(list(str(elem) if elem == "Iris-setosa" or elem == "Iris-versicolor" or elem == "Iris-virginica" else float(elem) for elem in row) for row in irisReader)

#----------------------------------------------------------------------------
# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 4), str, "V")

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# floating point operators
# Define a protected division function
def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)

# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

# Define a new in_range function
def in_range(left, right, input):
    if left >= input and input <= right:
        return True
    else:
        return False

# Define a new out_range function
def out_range(left, right, input):
    if left < input or input > right:
        return True
    else:
        return False

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(operator.ge, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, str, str], str)
pset.addPrimitive(in_range, [float, float, float], bool)
pset.addPrimitive(out_range, [float, float, float], bool)

# terminals
pset.addEphemeralConstant("rand80", lambda: random.random() * 80, float)
pset.addTerminal(True, bool)
pset.addTerminal(False, bool)
pset.addTerminal("Iris-virginica", str)
pset.addTerminal("Iris-versicolor", str)
pset.addTerminal("Iris-setosa", str)

#----------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalIrisbase(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    #Fitness function
    result = sum(str(func(*elem[:4])) == str(elem[4]) for elem in iris)
    return result,
    
toolbox.register("evaluate", evalIrisbase)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    random.seed(10)
    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("var", numpy.var)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats, halloffame=hof)
    
    return pop, stats, hof
```


```python
pop, stats, hof = main()
```

    gen	nevals	avg	std    	var	min	max
    0  	500   	50 	6.32456	40 	0  	100
    1  	303   	50.3	4.99099	24.91	0  	100
    2  	309   	50.558	7.1113 	50.5706	0  	100
    3  	316   	51.2  	7.65245	58.56  	50 	100
    4  	306   	52.7  	12.1536	147.71 	0  	100
    5  	296   	55.7  	15.8906	252.51 	50 	100
    6  	289   	61.22 	22.0662	486.916	0  	100
    7  	318   	67.712	26.0553	678.881	0  	100
    8  	282   	76.696	25.9393	672.848	0  	100
    9  	296   	84.13 	24.0332	577.593	0  	100
    10 	309   	85.67 	24.292 	590.101	0  	100
    11 	301   	86.724	23.7791	565.448	0  	100
    12 	298   	87.686	24.1839	584.863	0  	100
    13 	316   	88.062	22.9337	525.954	0  	100
    14 	291   	89.288	21.2307	450.741	0  	100
    15 	303   	91.012	20.0526	402.108	0  	100
    16 	272   	91.048	20.0253	401.014	0  	124
    17 	306   	91.016	20.7874	432.116	0  	124
    18 	306   	91.446	20.4376	417.695	0  	124
    19 	307   	89.366	23.6602	559.804	0  	124
    20 	299   	92.2  	21.232 	450.796	0  	124
    21 	314   	94.096	23.1825	537.427	0  	124
    22 	300   	100.078	22.3685	500.352	0  	124
    23 	315   	105.932	21.7811	474.415	3  	124
    24 	310   	111.01 	22.0316	485.39 	0  	124
    25 	300   	112.888	20.5513	422.355	0  	124
    26 	317   	113.444	20.765 	431.187	0  	129
    27 	309   	113.544	19.8461	393.868	0  	129
    28 	303   	114.296	19.4378	377.828	0  	129
    29 	312   	113.538	20.3115	412.557	0  	129
    30 	307   	114.636	20.0881	403.532	0  	129
    31 	261   	116.442	19.1732	367.611	0  	129
    32 	288   	117.508	17.4181	303.39 	10 	129
    33 	296   	115.794	20.7333	429.872	0  	131
    34 	312   	117.1  	18.0812	326.93 	0  	131
    35 	302   	118.944	18.6493	347.797	0  	136
    36 	288   	119.474	19.7131	388.605	0  	136
    37 	303   	119.532	18.7798	352.681	50 	136
    38 	314   	121.526	17.3882	302.349	49 	136
    39 	305   	120.124	20.6682	427.173	0  	136
    40 	303   	120.64 	19.2404	370.194	49 	141
    41 	301   	122.584	18.8332	354.691	0  	141
    42 	291   	123.528	20.1944	407.813	0  	141
    43 	313   	126.022	19.2337	369.934	33 	141
    44 	329   	127.108	19.9348	397.396	27 	141
    45 	307   	127.79 	21.7229	471.886	0  	141
    46 	299   	127.414	23.0319	530.467	0  	143
    47 	282   	129.098	22.8615	522.648	0  	143
    48 	291   	131.922	21.6589	469.108	0  	143
    49 	303   	129.604	23.3651	545.927	48 	144
    50 	301   	130.244	22.0732	487.224	50 	144
    51 	295   	132.66 	19.4572	378.584	50 	144
    52 	308   	131.464	23.7955	566.225	0  	145
    53 	267   	133.306	20.7679	431.304	50 	145
    54 	285   	132.506	22.4294	503.078	0  	145
    55 	308   	131.608	24.0279	577.342	43 	145
    56 	305   	129.91 	25.833 	667.342	0  	145
    57 	292   	131.396	24.3856	594.655	0  	145
    58 	321   	132.41 	23.3977	547.454	45 	145
    59 	327   	132.206	24.2607	588.584	0  	145
    60 	298   	134.316	23.9437	573.3  	0  	145
    61 	317   	135.42 	20.5512	422.352	50 	145
    62 	276   	137.902	19.8716	394.88 	5  	145
    63 	295   	134.156	23.0931	533.292	45 	146
    64 	300   	136.36 	20.074 	402.966	45 	145
    65 	301   	134.86 	22.9612	527.216	0  	145
    66 	323   	133.968	23.2889	542.375	29 	145
    67 	309   	137.224	19.526 	381.266	50 	145
    68 	299   	136.782	19.3074	372.774	50 	145
    69 	305   	136.062	20.5375	421.79 	50 	145
    70 	288   	139.238	16.6652	277.729	50 	145
    71 	294   	139.532	17.0245	289.833	50 	145
    72 	276   	137.888	18.8375	354.851	50 	145
    73 	315   	137.918	19.3985	376.303	38 	145
    74 	323   	136.938	20.9745	439.93 	45 	145
    75 	299   	139.02 	17.9249	321.304	0  	145
    76 	303   	138.66 	18.6805	348.96 	45 	145
    77 	296   	139.644	16.597 	275.461	45 	145
    78 	295   	139.532	17.1707	294.833	50 	145
    79 	301   	138.766	17.8632	319.095	23 	145
    80 	282   	140.22 	16.1611	261.18 	50 	145
    81 	296   	140.458	15.0607	226.824	50 	145
    82 	286   	140.928	14.2605	203.363	50 	145
    83 	296   	139.464	17.1943	295.645	50 	145
    84 	296   	139.166	18.3035	335.018	45 	145
    85 	286   	139.482	17.1982	295.778	50 	146
    86 	292   	140.28 	15.7679	248.626	50 	146
    87 	294   	140.146	16.1895	262.101	43 	146
    88 	302   	141.386	13.3888	179.261	50 	146
    89 	317   	140.06 	16.7683	281.176	0  	146
    90 	311   	140.022	16.0213	256.682	50 	146
    91 	282   	139.648	17.409 	303.072	50 	146
    92 	317   	139.466	17.5521	308.077	50 	146
    93 	316   	138.848	18.9211	358.009	0  	146
    94 	312   	135.344	22.2756	496.202	38 	147
    95 	291   	137.224	20.7057	428.726	0  	147
    96 	305   	137.272	21.4155	458.622	0  	147
    97 	292   	136.778	20.7205	429.341	0  	147
    98 	321   	138.094	18.9253	358.165	50 	147
    99 	323   	138.968	16.819 	282.879	50 	147
    100	306   	137.552	19.8115	392.495	50 	147



```python
print("Hall of fame")
print("")
print(str(hof[0])) #print the best individual which has the best fitness
```

    Hall of fame
    
    if_then_else(in_range(V1, V2, V1), if_then_else(in_range(V1, V2, V1), if_then_else(True, if_then_else(gt(add(V0, V3), add(mul(V3, V2), sub(V1, V1))), if_then_else(True, if_then_else(in_range(V1, V2, V1), 'Iris-versicolor', if_then_else(True, 'Iris-setosa', 'Iris-virginica')), 'Iris-versicolor'), if_then_else(in_range(V1, sub(add(44.637567096109805, V1), mul(2.0401660075381134, V1)), V1), if_then_else(False, if_then_else(in_range(V2, sub(add(44.637567096109805, protectedDiv(V2, add(V0, V3))), mul(V0, V1)), V1), if_then_else(in_range(V1, V2, V1), 'Iris-versicolor', 'Iris-virginica'), 'Iris-virginica'), if_then_else(in_range(V1, sub(add(44.637567096109805, V2), mul(V0, protectedDiv(add(V2, V2), sub(V1, V3)))), V1), 'Iris-versicolor', 'Iris-virginica')), 'Iris-virginica')), 'Iris-virginica'), 'Iris-virginica'), if_then_else(ge(mul(mul(V1, V3), V1), add(15.496936405952813, V2)), 'Iris-setosa', if_then_else(True, 'Iris-setosa', 'Iris-virginica')))


## Function to draw decision trees


```python
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

def plotting(individual):
    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
```


```python
plotting(hof[0]) #plotting the best individual
```

![Figure_1.png](attachment:Figure_1.png)
