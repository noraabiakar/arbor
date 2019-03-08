#include <arbor/common_types.hpp>

#include <string.h>
#include <stdio.h>
#include <hdf5.h>
#include <assert.h>

#define MAX_NAME 1024

using arb::cell_size_type;

class h5_dataset {
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

    int num_elements() {
        return size_;
    }

    /*auto get_value_at(hid_t i) {
        hid_t ds_id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
        int dset_data
        H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
        H5Dclose(ds_id);
    }*/

private:
    hid_t parent_id_;
    hid_t id_;
    std::string name_;
    size_t size_;
};

class h5_group {
public:

    h5_group(hid_t parent, std::string name): parent_id_(parent), name_(name), group_h_(parent_id_, name_) {

        hsize_t nobj;
        H5Gget_num_objs(group_h_.id, &nobj);

        char memb_name[MAX_NAME];

        groups_.reserve(nobj);

        for (unsigned i = 0; i < nobj; i++) {
            H5Gget_objname_by_idx(group_h_.id, (hsize_t)i, memb_name, (size_t)MAX_NAME);
            hid_t otype = H5Gget_objtype_by_idx(group_h_.id, (size_t)i);
            if (otype == H5G_GROUP) {
                groups_.emplace_back(group_h_.id, memb_name);
            }
            else if (otype == H5G_DATASET) {
                h5_dataset h(group_h_.id, memb_name);
                datasets_.push_back(std::move(h));
            }
        }
    }

    ~h5_group() {
        //std::cout << "destroying group "<< name_ << std::endl;
    }

    std::string name() {
        return name_;
    }

private:
    struct group_handle {
        group_handle(hid_t parent_id, std::string name): id(H5Gopen(parent_id, name.c_str(), H5P_DEFAULT)), name(name){
            //std::cout << "group handle " << id << " " << name << ": con\n";
        }
        ~group_handle() {
            H5Gclose(id);
            //std::cout << "group handle " << id << " " << name << ": des\n";
        }
        hid_t id;
        std::string name;
    };

    hid_t parent_id_;
    std::string name_;
    group_handle group_h_;

public:
    std::vector<h5_group> groups_;
    std::vector<h5_dataset> datasets_;
};



class h5_file {
public:
    h5_file(std::string name): file_(name), file_h_(name), top_group_(file_h_.id, "/"){}

    ~h5_file() {
        //std::cout << "destroying file "<< file_ << std::endl;
    }

    std::string get_name() {
        return file_;
    }

private:
    struct file_handle {
        file_handle(std::string file): id(H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT)), name(file) {
            //std::cout << "file handle " << id << " " << name << ": con\n";
        }
        ~file_handle() {
            H5Fclose(id);
            //std::cout << "file handle " << id << " " << name <<  ": des\n";
        }
        hid_t id;
        std::string name;
    };

    std::string file_;
    file_handle file_h_;

public:
    h5_group top_group_;
};
