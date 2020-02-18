#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include <hdf5.h>

/// Class for reading from hdf5 datasets
/// Datasets are opened and closed every time they are read
class h5_dataset {
public:
    // Constructor from parent (hdf5 group) id and dataset name - finds size of the dataset
    h5_dataset(hid_t parent, std::string name);

    // Constructor from parent (hdf5 group) id and dataset name - creates andf writes dataset `data`
    h5_dataset(hid_t parent, std::string name, std::vector<int> data);

    h5_dataset(hid_t parent, std::string name, std::vector<double> data);

    h5_dataset(hid_t parent, std::string name, std::vector<std::vector<int>> data);

    h5_dataset(hid_t parent, std::string name, std::vector<std::vector<double>> data);

    // returns name of dataset
    std::string name();

    // returns number of elements in a dataset
    int size();

    // Read all dataset
    template <typename T>
    auto get();

    // Read at index `i`; throws exception if out of bounds
    template <typename T>
    auto get(const int i);

    // Read range  between indices `i` and `j`; throws exception if out of bounds
    template <typename T>
    auto get(const int i, const int j);

private:
    // id of parent group
    hid_t parent_id_;

    // name of dataset
    std::string name_;

    // First dimension of dataset
    size_t size_;
};


/// Class for keeping track of what's in an hdf5 group (groups and datasets with pointers to each)
class h5_group {
public:
    // Constructor from parent (hdf5 group) id and group name
    // Builds tree of groups, each with it's own sub-groups and datasets
    h5_group(hid_t parent, std::string name);

    // Returns name of group
    std::string name();

    // Add a new group
    std::shared_ptr<h5_group> add_group(std::string name);

    // Add a new int dataset
    template <typename T>
    void add_dataset(std::string name, std::vector<T> dset);

    const std::vector<std::shared_ptr<h5_group>>& groups() {
        return groups_;
    }

    const std::vector<std::shared_ptr<h5_dataset>>& datasets() {
        return datasets_;
    }

    void print_groups() {
        for (auto g: groups()) {
            std::cout << g->name() << std::endl;
            std::cout << "->" << std::endl;
            g->print_groups();
            std::cout << "<-" << std::endl;
        }
    }

private:
    // RAII to handle recursive opening/closing groups
    struct group_handle {
        group_handle(hid_t parent_id, std::string name): name(name) {
            if (H5Lexists(parent_id, name.c_str(), H5P_DEFAULT)) {
                id = H5Gopen(parent_id, name.c_str(), H5P_DEFAULT);
            } else {
                id = H5Gcreate(parent_id, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            }
        }
        ~group_handle() {
            H5Gclose(id);
        }
        hid_t id;
        std::string name;
    };

    // id of parent group
    hid_t parent_id_;

    // name of group
    std::string name_;

    // Handles group opening/closing
    group_handle group_h_;

    // hdf5 groups belonging to group
    std::vector<std::shared_ptr<h5_group>> groups_;

    // hdf5 datasets belonging to group
    std::vector<std::shared_ptr<h5_dataset>> datasets_;
};


/// Class for an hdf5 file, holding a pointer to the top level group in the file
class h5_file {
private:
    // RAII to handle opening/closing files
    struct file_handle {
        file_handle(std::string file, bool new_file = false): name(file) {
            if (new_file) {
                id = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            }
            else {
                id = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            }
        }
        ~file_handle() {
            H5Fclose(id);
        }
        hid_t id;
        std::string name;
    };

    // name of file
    std::string name_;

    // Handles file opening/closing
    file_handle file_h_;

    // Pointer to top level group in file
    std::shared_ptr<h5_group> top_group_;

public:
    // Constructor from file name
    h5_file(std::string name, bool new_file=false);

    // Returns file name
    std::string name();

    const auto& top_group() {
        return top_group_;
    }
};

/// Class that wraps an h5_group
/// Provides direct read access to datasets in the group
/// Provides access to sub-groups of the group
class h5_wrapper {
public:
    h5_wrapper();

    h5_wrapper(const std::shared_ptr<h5_group>& g);
    h5_wrapper(h5_file file): h5_wrapper(file.top_group()) {};

    // Returns number of sub-groups in the wrapped h5_group
    int size();

    // Returns index of sub-group with name `name`; returns -1 if sub-group not found
    int find_group(std::string name) const;

    // Returns index of dataset with name `name`; returns -1 if dataset not found
    int find_dataset(std::string name) const;

    // Returns size of dataset with name `name`; returns -1 if dataset not found
    int dataset_size(std::string name) const;

    // Returns value at index i of dataset with name `name`; throws exception if dataset not found
    template <typename T>
    T get(std::string name, unsigned i) const;

    // Returns values between indices i and j of dataset with name `name`; throws exception if dataset not found
    template <typename T>
    T get(std::string name, unsigned i, unsigned j) const;

    // Returns full content of 1D dataset with name `name`; throws exception if dataset not found
    template <typename T>
    T get(std::string name) const;

    // Returns h5_wrapper of group at index i in members_
    const h5_wrapper& operator[] (unsigned i) const;

    // Returns h5_wrapper of group with name `name` in members_
    const h5_wrapper& operator[] (std::string name) const;

    // Returns name of the wrapped h5_group
    std::string name() const;

private:
    // Pointer to the h5_group wrapped in h5_wrapper
    std::shared_ptr<h5_group> ptr_;

    // Vector of h5_wrappers around sub_groups of the h5_group
    std::vector<h5_wrapper> members_;

    // Map from name of dataset to index in vector of datasets in wrapped h5_group
    std::unordered_map<std::string, unsigned> dset_map_;

    // Map from name of sub-group to index in vector of sub-groups in wrapped h5_group
    std::unordered_map<std::string, unsigned> member_map_;
};

struct local_element{
    std::string pop_name;
    unsigned el_id;
};