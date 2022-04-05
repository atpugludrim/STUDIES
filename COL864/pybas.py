import pandas as pd # for data manipulation 
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs

# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

D = BbnNode(Variable(0,'D',[0,1]),[0.2,0.8])
C = BbnNode(Variable(5,'C',[0,1]),[0.7,0.3])
B = BbnNode(Variable(2,'B',[0,1]),[0.1,.9,.2,.8])
A = BbnNode(Variable(1,'A',[0,1]),[1,0,0,1,.1,.9,.2,.8])
bbn = Bbn().add_node(A).add_node(B).add_node(C).add_node(D).add_edge(Edge(D,B,EdgeType.DIRECTED)).add_edge(Edge(B,A,EdgeType.DIRECTED)).add_edge(Edge(C,A,EdgeType.DIRECTED))
join_tree = InferenceController.apply(bbn)
ev = EvidenceBuilder().with_node(join_tree.get_bbn_node_by_name('C')).with_evidence(0,1.0).build()
join_tree.set_observation(ev)
ev = EvidenceBuilder().with_node(join_tree.get_bbn_node_by_name('A')).with_evidence(1,1.0).build()
join_tree.set_observation(ev)
for node, posteriors in join_tree.get_posteriors().items():
    p = ', '.join([f'{val}={prob:.5f}' for val,prob in posteriors.items()])
    print(f'{node}: {p}')
# A = BbnNode(Variable(0,'A',['low','high']),[0.7,0.3])
# B = BbnNode(Variable(1,'B',['faulty','non-faulty']),[0.1,0.9,0.6,0.4])
# C = BbnNode(Variable(2,'C',['low','high']),[0.7,0.3,0.99,0.01,0.3,0.7,0.01,0.99])
# D = BbnNode(Variable(3,'D',['off','on']),[1,0,0.99,0.01,1,0,0.3,0.7])
# E = BbnNode(Variable(4,'E',['faulty','non-faulty']),[0.8,0.2])
# 
# bbn = Bbn().add_node(A).add_node(B).add_node(C).add_node(D).add_node(E)\
#         .add_edge(Edge(A,B,EdgeType.DIRECTED))\
#         .add_edge(Edge(A,C,EdgeType.DIRECTED))\
#         .add_edge(Edge(B,C,EdgeType.DIRECTED))\
#         .add_edge(Edge(C,D,EdgeType.DIRECTED))\
#         .add_edge(Edge(E,D,EdgeType.DIRECTED))
# 
# join_tree = InferenceController.apply(bbn)
# 
# ev = EvidenceBuilder().with_node(join_tree.get_bbn_node_by_name('B')).with_evidence('non-faulty',1.0).build()
# join_tree.set_observation(ev)
# 
# ev = EvidenceBuilder()\
#         .with_node(join_tree.get_bbn_node_by_name('E'))\
#         .with_evidence('non-faulty',1.0)\
#         .build()
# join_tree.set_observation(ev)
# 
# ev = EvidenceBuilder()\
#         .with_node(join_tree.get_bbn_node_by_name('D'))\
#         .with_evidence('on',1.0).build()
# join_tree.set_observation(ev)
# 
# for node, posteriors in join_tree.get_posteriors().items():
#     p = ', '.join([f'{val}={prob:.5f}' for val,prob in posteriors.items()])
#     print(f'{node}: {p}')
