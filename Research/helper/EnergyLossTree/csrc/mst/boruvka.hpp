// Adapted from megvii-research/TreeEnergyLoss (Apache-2.0).

#pragma once


struct Edge {
    int src;
    int dest;
    float weight;
};


struct Graph {
    int vertices;
    int edges;
    Edge* edge;
};


Graph* create_graph(int vertices, int edges);
void boruvka_mst(Graph* graph, int* edge_out);
