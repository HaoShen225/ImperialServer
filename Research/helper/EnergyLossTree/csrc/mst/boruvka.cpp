// Adapted from megvii-research/TreeEnergyLoss (Apache-2.0).
// The algorithm and deterministic strict-less-than tie handling are retained.

#include "boruvka.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>


namespace {

struct Subset {
    int parent;
    int rank;
};


int find_root(std::vector<Subset>& subsets, int vertex) {
    if (subsets[vertex].parent != vertex) {
        subsets[vertex].parent = find_root(subsets, subsets[vertex].parent);
    }
    return subsets[vertex].parent;
}


void union_sets(std::vector<Subset>& subsets, int left, int right) {
    const int left_root = find_root(subsets, left);
    const int right_root = find_root(subsets, right);
    if (subsets[left_root].rank < subsets[right_root].rank) {
        subsets[left_root].parent = right_root;
    } else if (subsets[left_root].rank > subsets[right_root].rank) {
        subsets[right_root].parent = left_root;
    } else {
        subsets[right_root].parent = left_root;
        subsets[left_root].rank += 1;
    }
}

}  // namespace


void boruvka_mst(Graph* graph, int* edge_out) {
    const int vertex_count = graph->vertices;
    const int edge_count = graph->edges;
    std::vector<Subset> subsets(vertex_count);
    std::vector<int> cheapest(vertex_count, -1);

    for (int vertex = 0; vertex < vertex_count; ++vertex) {
        subsets[vertex].parent = vertex;
        subsets[vertex].rank = 0;
    }

    int tree_count = vertex_count;
    int output_count = 0;
    while (tree_count > 1) {
        std::fill(cheapest.begin(), cheapest.end(), -1);
        for (int edge_index = 0; edge_index < edge_count; ++edge_index) {
            const Edge& edge = graph->edge[edge_index];
            const int first_set = find_root(subsets, edge.src);
            const int second_set = find_root(subsets, edge.dest);
            if (first_set == second_set) {
                continue;
            }
            if (cheapest[first_set] == -1 ||
                graph->edge[cheapest[first_set]].weight > edge.weight) {
                cheapest[first_set] = edge_index;
            }
            if (cheapest[second_set] == -1 ||
                graph->edge[cheapest[second_set]].weight > edge.weight) {
                cheapest[second_set] = edge_index;
            }
        }

        bool merged = false;
        for (int vertex = 0; vertex < vertex_count; ++vertex) {
            const int edge_index = cheapest[vertex];
            if (edge_index == -1) {
                continue;
            }
            const Edge& edge = graph->edge[edge_index];
            const int first_set = find_root(subsets, edge.src);
            const int second_set = find_root(subsets, edge.dest);
            if (first_set == second_set) {
                continue;
            }
            edge_out[2 * output_count] = edge.src;
            edge_out[2 * output_count + 1] = edge.dest;
            output_count += 1;
            union_sets(subsets, first_set, second_set);
            tree_count -= 1;
            merged = true;
        }
        if (!merged) {
            throw std::runtime_error("Boruvka MST received a disconnected graph.");
        }
    }
}


Graph* create_graph(int vertices, int edges) {
    Graph* graph = new Graph;
    graph->vertices = vertices;
    graph->edges = edges;
    graph->edge = new Edge[edges];
    return graph;
}
