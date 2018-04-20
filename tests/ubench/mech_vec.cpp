// Test performance of vectorization for mechanism implementations.
//
// Start with pas (passive dendrite) mechanism

#include <backends/multicore/fvm.hpp>
#include <benchmark/benchmark.h>
#include <fvm_lowered_cell_impl.hpp>

using namespace arb;

using backend = arb::multicore::backend;
using fvm_cell = arb::fvm_lowered_cell_impl<backend>;

class test_recipe: public recipe {
    unsigned num_comp_;
public:
    test_recipe(unsigned num_comp): num_comp_(num_comp) {}

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        cell c;

        auto soma = c.add_soma(12.6157/2.0);
        soma->add_mechanism("pas");

        c.add_cable(0, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);

        for (auto& seg: c.segments()) {
            if (seg->is_dendrite()) {
                seg->add_mechanism("pas");
                seg->set_compartments(num_comp_-1);
            }
        }
        return std::move(c);
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable1d_neuron;
    }

    //virtual cell_size_type num_targets(cell_gid_type) const { return 0; }
};

void pas_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    probe_association_map<probe_handle> probe_handles;
    test_recipe rec(ncomp);

    fvm_cell cell;
    cell.initialize(gids, rec, target_handles, probe_handles);

    int idx = -1;
    auto &mechs = cell.mechanisms();
    for (unsigned i=0; i<mechs.size(); ++i) {
        if (mechs[i]->internal_name() == "pas") {
            idx = i;
        }
    }
    if (idx==-1) {
        std::cout << "ERROR: couldn't find pas\n";
        exit(1);
    }
    auto& m = mechs[idx];

    while (state.KeepRunning()) {
        // call nrn_current
        m->nrn_current();
    }
}

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncomps: {10, 100, 1000, 10000, 100000, 1000000, 10000000}) {
        b->Args({ncomps});
    }
}

BENCHMARK(pas_current)->Apply(run_custom_arguments);
BENCHMARK_MAIN();
