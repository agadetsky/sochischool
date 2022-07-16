#include "ATen/Parallel.h"
#include <torch/extension.h>
#include <iostream>
#include <vector>

using namespace std;


class GEdge {
public:
    GEdge(int v1, int v2, double weight) {
        this->v1 = v1;
        this->v2 = v2;
        this->weight = weight;
    }
    int v1;
    int v2;
    double weight;
};


bool edgeCompare (GEdge a, GEdge b);

class GNode {
public:
    GNode * parent;
    int rank;
    int value;
};

class Union_Find {
private:
    vector<GNode *> sets; // this makes cleanup and testing much easier.
    GNode * link(GNode * a, GNode * b);
    
public:
    Union_Find(int size);
    Union_Find();
    void makeset(int x);
    void onion(int x, int y);
    GNode * find(int x);
    void clean();
    vector<GNode *> raw();
};


bool edgeCompare (GEdge a, GEdge b) { return (a.weight > b.weight); }

GNode * Union_Find::link(GNode * a, GNode * b) {
    // put the smaller rank tree as a child of the bigger rank tree.
    // otherwise (equal rank), put second element as parent.
    if (a->rank > b->rank) {
        // swap pointers
        GNode * temp_ptr = b;
        b = a;
        a = temp_ptr;
    }
    if (a->rank == b->rank) {
        // update the rank of the new parrent
        b->rank = b->rank + 1;
    }
    
    // a is child of b
    a->parent = b;
    return b;
}

Union_Find::Union_Find(int size) {
    // optimized init.
    sets.resize(size);
}

Union_Find::Union_Find() {}

void Union_Find::makeset(int x) {
    // takes in a vertex. creates a set out of it solo.
    
    GNode * n = new GNode();
    n->value = x;
    n->rank = 0;
    n->parent = n;
    
    if (sets.size() <= x) {
        sets.resize(x + 1); // +1 handles 0 index, but watch out for other issues.
        // Best to initialize with a suggested size.
    }
    sets[x] = n;
}

// "union" is taken
void Union_Find::onion(int x, int y) {
    // replace two sets containing x and y with their union.
    this->link(this->find(x), this->find(y));
}

GNode * Union_Find::find(int x) {
    GNode * n = sets[x];
    
    if (n->parent->value != n->value) {
        // walk the node up the tree (flattens as it finds)
        n->parent = find(n->parent->value);
    }
    
    return n->parent;
}

void Union_Find::clean() {
    // Normally I would just make a destructor,
    // but scoping is strange with iterative deepening.
    
    for(int i = 0; i < sets.size(); i++) {
        free(sets[i]);
    }
    sets.clear();
}

vector<GNode *> Union_Find::raw() {
    return sets;
};

torch::Tensor kruskals(torch::Tensor weights, torch::Tensor lengths) {
    int n = weights.size(-1);
    int batch_size = weights.size(0);
    torch::Tensor adj_matrix = torch::zeros({batch_size, n, n});
    for (int sample_idx=0; sample_idx < batch_size; sample_idx++) {
        int length = lengths[sample_idx].item<int>();
        Union_Find uf;
        vector<GEdge> edges;
        for (int src = 0; src < length; src++) {
            for (int dst = src + 1;  dst < length; dst++) {
                double w = weights[sample_idx][src][dst].item<double>();
                // Only consider vertices that are actually included in that
                // batch, given by lengths.
                GEdge e(src, dst, w);
                edges.push_back(e);
            }
            // Insert vertices into union find.
            uf.makeset(src);
            if (src == (length - 1)){
                uf.makeset(src + 1);
            }
        }
        
        sort(edges.begin(), edges.end(), edgeCompare);

        int mst_count = 0;
        for(int i = 0; i < edges.size(); i++) {
            GEdge e = edges[i];
            if (uf.find(e.v1) != uf.find(e.v2)) {
                uf.onion(e.v1, e.v2);
                mst_count = mst_count + 1;

                // Update adjacency matrix.
                adj_matrix[sample_idx][e.v1][e.v2] = 1;
                adj_matrix[sample_idx][e.v2][e.v1] = 1;

                if (mst_count >= (lengths[sample_idx].item<int>() - 1)) { // |V| - 1 edges
                    break;
                }
            }
        }
        uf.clean();
    }
    return adj_matrix;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kruskals", &kruskals, "Kruskals");
}