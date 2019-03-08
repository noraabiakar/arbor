#include <arbor/common_types.hpp>
#include <arbor/util/optional.hpp>
#include <string.h>
#include <stdio.h>
#include <hdf5.h>
#include <assert.h>

#define MAX_NAME 1024

using arb::cell_size_type;

class h5_dataset {
private:
    hid_t parent_id_;
    hid_t id_;
    std::string name_;
    size_t size_;

public:
    h5_dataset(hid_t parent, std::string name): parent_id_(parent), name_(name) {
        id_ = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
        hid_t dspace = H5Dget_space(id_);

        const int ndims = H5Sget_simple_extent_ndims(dspace);
        if (ndims > 1) {
            std::cout << "Dataset: " << name_ << " is multidimensional\n";
        }

        hsize_t dims[ndims];
        H5Sget_simple_extent_dims(dspace, dims, NULL);

        size_ = dims[0];

        H5Dclose(id_);
    }

    ~h5_dataset() {
    }

    std::string name() {
        return name_;
    }

    int size() {
        return size_;
    }

    auto at(const int i) {
        const hsize_t idx = (hsize_t)i;

        // Output
        int *out = new int[1];

        // Output dimensions 1x1
        hsize_t dims = 1;
        hsize_t dim_sizes[] = {1};

        // Output size
        hsize_t num_elements = 1;

        id_ = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
        hid_t dspace = H5Dget_space(id_);

        H5Sselect_elements(dspace, H5S_SELECT_SET, num_elements, &idx);
        hid_t out_mem = H5Screate_simple(dims, dim_sizes, NULL);

        H5Dread(id_, H5T_NATIVE_INT, out_mem, dspace, H5P_DEFAULT, out);
        H5Dclose(id_);

        return out[0];
    }

    auto at(const int i, const int j) {
        const hsize_t idx[2] = {(hsize_t)i, (hsize_t)j};

        // Output
        int *out = new int[1];

        // Output dimensions 1x1
        hsize_t dims = 1;
        hsize_t dim_sizes[] = {1};

        // Output size
        hsize_t num_elements = 1;

        id_ = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
        hid_t dspace = H5Dget_space(id_);

        H5Sselect_elements(dspace, H5S_SELECT_SET, num_elements, idx);
        hid_t out_mem = H5Screate_simple(dims, dim_sizes, NULL);

        H5Dread(id_, H5T_NATIVE_INT, out_mem, dspace, H5P_DEFAULT, out);
        H5Dclose(id_);

        return out[0];
    }

    auto all() {
        int out_a[size_];
        id_ = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);

        H5Dread(id_, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                out_a);
        H5Dclose(id_);

        std::vector<int> out(out_a, out_a + size_);

        return out;
    }
};

class h5_group {
private:
    struct group_handle {
        group_handle(hid_t parent_id, std::string name): id(H5Gopen(parent_id, name.c_str(), H5P_DEFAULT)), name(name){}
        ~group_handle() {
            H5Gclose(id);
        }
        hid_t id;
        std::string name;
    };

    hid_t parent_id_;
    std::string name_;
    group_handle group_h_;

public:
    std::vector<std::shared_ptr<h5_group>> groups_;
    std::vector<std::shared_ptr<h5_dataset>> datasets_;

    h5_group(hid_t parent, std::string name): parent_id_(parent), name_(name), group_h_(parent_id_, name_) {

        hsize_t nobj;
        H5Gget_num_objs(group_h_.id, &nobj);

        char memb_name[MAX_NAME];

        groups_.reserve(nobj);

        for (unsigned i = 0; i < nobj; i++) {
            H5Gget_objname_by_idx(group_h_.id, (hsize_t)i, memb_name, (size_t)MAX_NAME);
            hid_t otype = H5Gget_objtype_by_idx(group_h_.id, (size_t)i);
            if (otype == H5G_GROUP) {
                groups_.emplace_back(std::make_shared<h5_group>(group_h_.id, memb_name));
            }
            else if (otype == H5G_DATASET) {
                datasets_.emplace_back(std::make_shared<h5_dataset>(group_h_.id, memb_name));
            }
        }
    }

    std::string name() {
        return name_;
    }
};

class h5_file {
private:
    struct file_handle {
        file_handle(std::string file): id(H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT)), name(file) {}
        ~file_handle() {
            H5Fclose(id);
        }
        hid_t id;
        std::string name;
    };

    std::string file_;
    file_handle file_h_;

public:
    std::shared_ptr<h5_group> top_group_;
    h5_file(std::string name):
            file_(name),
            file_h_(name),
            top_group_(std::make_shared<h5_group>(file_h_.id, "/")) {}

    std::string name() {
        return file_;
    }
};

///////////////////////////////////////////////////////////

class dataspace {
public:
    dataspace(const std::shared_ptr<h5_group>& g): ptr(g) {
        unsigned i = 0;
        for (auto d: ptr->datasets_) {
            dset_map[d->name()] = i++;
        }
        i = 0;
        for (auto g: ptr->groups_) {
            member_map[g->name()] = i++;
            members.emplace_back(g);
        }
    }

    int find_group(std::string name) {
        if (member_map.find(name) != member_map.end()) {
            return member_map[name];
        }
        return -1;
    }

    int find_dataset(std::string name) {
        if (dset_map.find(name) != dset_map.end()) {
            return dset_map[name];
        }
        return -1;
    }

    arb::util::optional<int> dataset_at(std::string name, unsigned i) {
        if (find_dataset(name) != -1) {
            return ptr->datasets_[dset_map[name]]->at(i);
        }
        return arb::util::nullopt;
    }

    arb::util::optional<int> dataset_at(std::string name, unsigned i, unsigned j) {
        if (find_dataset(name)!= -1) {
            return ptr->datasets_[dset_map[name]]->at(i, j);
        }
        return arb::util::nullopt;
    }

    arb::util::optional<std::vector<int>> dataset(std::string name) {
        if (find_dataset(name)!= -1) {
            return ptr->datasets_[dset_map[name]]->all();
        }
        return arb::util::nullopt;
    }

    dataspace operator [](int i) const {
        return members[i];
    }

    arb::util::optional<dataspace> operator [](std::string name) {
        if (find_group(name) != -1) {
            return members[find_group(name)];
        }
        return arb::util::nullopt;
    }

private:
    std::shared_ptr<h5_group> ptr;
    std::vector<dataspace> members;
    std::unordered_map<std::string, unsigned> dset_map;
    std::unordered_map<std::string, unsigned> member_map;
};


class record {
public:
    record(const std::shared_ptr<h5_file>& file) {
        if (file->top_group_->groups_.size() != 1) {
            throw arb::sonata_file_exception("file hierarchy wrong\n");
        }

        for (auto g: file->top_group_->groups_.front()->groups_) {
            populations_.emplace_back(g);
        }

        for (auto& p: file->top_group_->groups_.front()->groups_) {
            for(auto& d: p->datasets_) {
                if(d->name().find("type_id") != std::string::npos) {
                    partition_.push_back(d->size());
                    num_elements_ += d->size();
                }
            }
        }
    }

    std::vector<dataspace> populations() {
        return populations_;
    }

    std::vector<cell_size_type> partitions() {
        return partition_;
    }

    int num_elements() {
        return num_elements_;
    }

    dataspace operator [](int i) const {
        return populations_[i];
    }

private:
    int num_elements_ = 0;
    std::vector<cell_size_type> partition_;
    std::vector<dataspace> populations_;
};