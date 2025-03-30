import rdkit
import networkx as nx
import networkit as nk


def main():
    graphs = nx.read_graph6("datasets/example_graph.g6")
    graph_nk = nk.nxadapter.nx2nk(graphs[0])

    deg = nk.centrality.DegreeCentrality(graph_nk ).run()
    print("Degree Centrality:", deg.scores())

    # Compute clustering coefficient
    clust = nk.centrality.LocalClusteringCoefficient(graph_nk ).run()
    print("Clustering Coefficient:", clust.scores())

    # Compute betweenness centrality
    bet = nk.centrality.Betweenness(graph_nk ).run()
    print("Betweenness Centrality:", bet.scores())

    # Compute PageRank
    pr = nk.centrality.PageRank(graph_nk, damp=0.85).run()
    print("PageRank Scores:", pr.scores())


if __name__ == "__main__":
    main()
