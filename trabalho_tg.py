import matplotlib.pyplot as plt
import networkx as nx

G1 = nx.erdos_renyi_graph(100, 0.05)

G2 = nx.watts_strogatz_graph(100, 4, 0.2)

G3 = nx.barabasi_albert_graph(100, 3)

nx.draw(G2, node_size = 30)
plt.title("erdos_renyi_graph")
plt.show()
nx.draw(G1, node_size = 30)
plt.title("watts_strogatz_graph")
plt.show()
nx.draw(G3, node_size = 30)
plt.title("barabasi_albert_graph")
plt.show()
