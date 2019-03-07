#include <arbor/common_types.hpp>

#include <string.h>
#include <stdio.h>
#include <hdf5.h>
#include <assert.h>

#define MAX_NAME 1024


using arb::cell_size_type;


std::vector<cell_size_type> get_population_patition(std::string filename);
void scan_group(hid_t, std::vector<cell_size_type>&);

std::vector<cell_size_type> get_population_patition(std::string filename) {
    std::vector<cell_size_type> partition;
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t grp = H5Gopen(file, "/", H5P_DEFAULT);
    scan_group(grp, partition);
    H5Fclose(file);
    return partition;
}

void scan_group(hid_t gid, std::vector<cell_size_type>& partition) {
    char memb_name[MAX_NAME];

    hsize_t nobj;
    H5Gget_num_objs(gid, &nobj);

    for (unsigned i = 0; i < nobj; i++) {
        H5Gget_objname_by_idx(gid, (hsize_t)i, memb_name, (size_t)MAX_NAME );
        int otype =  H5Gget_objtype_by_idx(gid, (size_t)i );

        switch(otype) {
            case H5G_GROUP:
            {
                hid_t grp_id = H5Gopen(gid, memb_name, H5P_DEFAULT);
                scan_group(grp_id, partition);
                H5Gclose(grp_id);
                break;
            }
            case H5G_DATASET:
            {
                hid_t ds_id = H5Dopen(gid, memb_name, H5P_DEFAULT);
                if (strcmp(memb_name, "node_type_id") == 0) {
                    hid_t dspace = H5Dget_space(ds_id);
                    const int ndims = H5Sget_simple_extent_ndims(dspace);
                    if (ndims > 1) {
                        printf("Too many entries\n");
                        exit(1);
                    }

                    hsize_t dims[ndims];
                    H5Sget_simple_extent_dims(dspace, dims, NULL);

                    partition.push_back(dims[0]);
                }
                H5Dclose(ds_id);
                break;
            }
            default: break;
        }
    }
}


class h5_dataset {
public:
    std::string get_name() {
        return name_;
    }

    int get_num_elements() {
        return size_;
    }

    void get_value_at(hid_t i) {
        hid_t file = H5Fopen(file_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t ds_id = H5Dopen(parent_, name_.c_str(), H5P_DEFAULT);
        H5Dclose(ds_id);
        H5Fclose(file);
    }
    friend class h5_group;

private:
    h5_dataset(hid_t parent, std::string name, std::string file): parent_(parent), name_(name), file_(file) {
        hid_t ds_id = H5Dopen(parent_, name_.c_str(), H5P_DEFAULT);
        hid_t dspace = H5Dget_space(ds_id);

        const int ndims = H5Sget_simple_extent_ndims(dspace);
        if (ndims > 1) {
            std::cout << "Dataset: " << name_ << " is multidimensional\n";
        }

        hsize_t dims[ndims];
        H5Sget_simple_extent_dims(dspace, dims, NULL);

        size_ = dims[0];

        H5Dclose(ds_id);
    }

    hid_t parent_;
    std::string name_;
    std::string file_;
    size_t size_;
};

class h5_group {
public:

    std::string get_name() {
        return name_;
    }

    int get_num_datasets() {
        return datasets_.size();
    }

    std::vector<h5_dataset> get_datasets() {
        return datasets_;
    }

    int get_num_groups() {
        return groups_.size();
    }

    std::vector<h5_group> get_groups() {
        return groups_;
    }

    friend class h5_file;

private:
    h5_group(hid_t parent, std::string name, std::string file): parent_(parent), name_(name), file_(file) {
        hid_t grp_id = H5Gopen(parent_, name_.c_str(), H5P_DEFAULT);

        hsize_t nobj;
        H5Gget_num_objs(grp_id, &nobj);

        char memb_name[MAX_NAME];

        for (unsigned i = 0; i < nobj; i++) {
            H5Gget_objname_by_idx(grp_id, (hsize_t)i, memb_name, (size_t)MAX_NAME);
            int otype = H5Gget_objtype_by_idx(grp_id, (size_t) i);
            if (otype == H5G_GROUP) {
                h5_group h(grp_id, memb_name, file_);
                groups_.push_back(std::move(h));
            }
            else if (otype == H5G_DATASET) {
                h5_dataset h(grp_id, memb_name, file_);
                datasets_.push_back(std::move(h));
            }
        }

        H5Gclose(grp_id);
    }

    hid_t parent_;
    std::string name_;
    std::string file_;
    std::vector<h5_group> groups_;
    std::vector<h5_dataset> datasets_;
};

class h5_file {
public:
    h5_file(std::string name): file_(name) {
        hid_t file = H5Fopen(file_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t id = H5Gopen(file, "/", H5P_DEFAULT);

        hsize_t nobj;
        H5Gget_num_objs(id, &nobj);

        if (nobj > 1) {
            std::cout << "Too many groups in the top level" << std::endl;
            exit(1);
        }

        // Open the first group in the top level: Should be either "nodes" or "edges"
        char memb_name[MAX_NAME];
        H5Gget_objname_by_idx(id, (hsize_t)0, memb_name, (size_t)MAX_NAME);
        hid_t top_id = H5Gopen(id, memb_name, H5P_DEFAULT);

        // Groups inside first group are the populations
        H5Gget_num_objs(top_id, &nobj);
        pops_.reserve(nobj);

        for (unsigned i = 0; i < nobj; i++) {
            H5Gget_objname_by_idx(top_id, (hsize_t)i, memb_name, (size_t)MAX_NAME);
            int otype = H5Gget_objtype_by_idx(top_id, (size_t) i);
            if (otype == H5G_GROUP) {
                h5_group h(top_id, memb_name, file_);
                pops_.push_back(std::move(h));
            }
        }

        num_elements_ = 0;
        partition_.reserve(pops_.size());
        for (auto p: pops_) {
            for (auto d: p.get_datasets()) {
                if (d.get_name().find("type_id") != std::string::npos) {
                    num_elements_ += d.get_num_elements();
                    partition_.push_back(d.get_num_elements());
                }
            }
        }

        H5Fclose(file);
    }

    std::string get_name() {
        return file_;
    }

    int get_num_populations() {
        return pops_.size();
    }

    std::vector<h5_group> get_populations() {
        return pops_;
    }

    int get_num_elements() {
        return num_elements_;
    }

    std::vector<size_t> get_partition() {
        return partition_;
    }

private:
    std::string file_;
    std::vector<h5_group> pops_;
    size_t num_elements_;
    std::vector<size_t> partition_;
};
