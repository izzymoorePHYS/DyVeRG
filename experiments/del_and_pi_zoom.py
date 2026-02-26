from os import getcwd
from os.path import join
import sys
sys.path.extend(['.', '..'])

import git
import numpy as np
import networkx as nx
from tqdm import trange
# from loguru import logger

# from dyverg.LightMultiGraph import convert
from src.data import load_data
from src.decomposition import decompose
from src.adjoin_graph import update_grammar
from src.utils import mkdir

if __name__ == '__main__':
    g = nx.karate_club_graph()
    print("karate club: n, ",g.number_of_nodes(), "; v, ", g.number_of_edges())
    #nx.draw(g, with_labels=True)
    #h = nx.wheel_graph(34)
    h = nx.karate_club_graph()
    #print(h.number_of_nodes(), ", ", h.number_of_edges())
    print("interior addition of one edge between (8)<->(9) at next time step")
    h.add_edge(8,9)
    print("karate club w/ (8)<->(9): n, ", h.number_of_nodes(), "; v, ", h.number_of_edges())
    #nx.draw(h, with_labels=True)
    grammar_g = decompose(g, time=1)
    grammar_h = decompose(h, time=2)
    print(grammar_g)
    print(grammar_h)
    sample_grammar = update_grammar(grammar_g, g, h, t1=1, t2=2, verbose=True)
    print(sample_grammar)
    print(sample_grammar.ll(1,2))
    print("\n\n~~ Repeating with more intense additions ~~\n")
    g = nx.karate_club_graph()
    print("karate club: n, ",g.number_of_nodes(), "; v, ", g.number_of_edges())
    h1 = nx.karate_club_graph()
    #print(h1.number_of_nodes(), ", ", h1.number_of_edges())
    print("interior addition of three edges between (8)<->(9),(15)<->(30), and (5)<->(30) at next time step")
    h1.add_edge(8, 9)
    h1.add_edge(15, 30)
    h1.add_edge(5, 30)
    print("karate club w/ (8)<->(9),(15)<->(30), and (5)<->(30): n, ",h1.number_of_nodes(), "; v, ", h1.number_of_edges())
    grammar_g = decompose(g, time=1)
    grammar_h1 = decompose(h1, time=2)
    print(grammar_g)
    print(grammar_h1)
    sample_grammar = update_grammar(grammar_g, g, h1, t1=1, t2=2, verbose=True)
    print(sample_grammar)
    print(sample_grammar.ll(1, 2))




