#include <iostream>
#include <string>
#include <stdexcept>

#include <hdf5.h>

#include "include/hdf5_lib.hpp"

#define MAX_NAME 1024

///h5_dataset methods

h5_dataset::h5_dataset(hid_t parent, std::string name): parent_id_(parent), name_(name) {
    auto id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
    hid_t dspace = H5Dget_space(id);

    const int ndims = H5Sget_simple_extent_ndims(dspace);

    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dspace, dims, NULL);

    size_ = dims[0];

    H5Sclose(dspace);
    H5Dclose(id);
}

h5_dataset::h5_dataset(hid_t parent, std::string name, std::vector<int> data): parent_id_(parent), name_(name) {
    hsize_t size = data.size();
    auto dspace = H5Screate_simple(1, &size, NULL);

    auto id = H5Dcreate(parent_id_, name.c_str(), H5T_NATIVE_INT, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int arr[size];
    std::copy(data.begin(), data.end(), arr);
    H5Dwrite(id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr);

    H5Sclose(dspace);
    H5Dclose(id);
}

h5_dataset::h5_dataset(hid_t parent, std::string name, std::vector<double> data): parent_id_(parent), name_(name) {
    hsize_t size = data.size();
    auto dspace = H5Screate_simple(1, &size, NULL);

    auto id = H5Dcreate(parent_id_, name.c_str(), H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    double arr[size];
    std::copy(data.begin(), data.end(), arr);
    H5Dwrite(id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr);

    H5Sclose(dspace);
    H5Dclose(id);
}

h5_dataset::h5_dataset(hid_t parent, std::string name, std::vector<std::vector<int>> data): parent_id_(parent), name_(name) {
    hsize_t dims_data[2] = {data.size(), data.front().size()};
    auto dspace = H5Screate_simple(2, dims_data, NULL);

    auto id = H5Dcreate(parent_id_, name.c_str(), H5T_NATIVE_INT, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int arr[data.size()][data.front().size()];
    for (unsigned i = 0; i < data.size(); i++) {
        std::copy(data[i].begin(), data[i].end(), arr[i]);
    }
    H5Dwrite(id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr);

    H5Sclose(dspace);
    H5Dclose(id);
}

h5_dataset::h5_dataset(hid_t parent, std::string name, std::vector<std::vector<double>> data): parent_id_(parent), name_(name) {
    hsize_t dims_data[2] = {data.size(), data.front().size()};
    auto dspace = H5Screate_simple(2, dims_data, NULL);

    auto id = H5Dcreate(parent_id_, name.c_str(), H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    double arr[data.size()][data.front().size()];
    for (unsigned i = 0; i < data.size(); i++) {
        std::copy(data[i].begin(), data[i].end(), arr[i]);
    }
    H5Dwrite(id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr);

    H5Sclose(dspace);
    H5Dclose(id);
}

std::string h5_dataset::name() {
    return name_;
}

int h5_dataset::size() {
    return size_;
}

template<>
auto h5_dataset::get<int>(const int i) {
    const hsize_t idx = (hsize_t)i;

    // Output
    int *out = new int[1];

    // Output dimensions 1x1
    hsize_t dims = 1;
    hsize_t dim_sizes[] = {1};

    // Output size
    hsize_t num_elements = 1;

    auto id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
    hid_t dspace = H5Dget_space(id);

    H5Sselect_elements(dspace, H5S_SELECT_SET, num_elements, &idx);

    hid_t out_mem = H5Screate_simple(dims, dim_sizes, NULL);

    auto status = H5Dread(id, H5T_NATIVE_INT, out_mem, dspace, H5P_DEFAULT, out);

    H5Sclose(dspace);
    H5Sclose(out_mem);
    H5Dclose(id);

    if (status < 0 ) {
        throw std::runtime_error("error reading dataset " + name() + " at " + std::to_string(i));
    }

    int r = out[0];
    delete [] out;

    return r;
}

template<>
auto h5_dataset::get<double>(const int i) {
    const hsize_t idx = (hsize_t)i;

    // Output
    double *out = new double[1];

    // Output dimensions 1x1
    hsize_t dims = 1;
    hsize_t dim_sizes[] = {1};

    // Output size
    hsize_t num_elements = 1;

    auto id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
    hid_t dspace = H5Dget_space(id);

    H5Sselect_elements(dspace, H5S_SELECT_SET, num_elements, &idx);
    hid_t out_mem = H5Screate_simple(dims, dim_sizes, NULL);

    auto status = H5Dread(id, H5T_NATIVE_DOUBLE, out_mem, dspace, H5P_DEFAULT, out);

    H5Sclose(dspace);
    H5Sclose(out_mem);
    H5Dclose(id);

    if (status < 0) {
        throw std::runtime_error("error reading dataset " + name() + " at " + std::to_string(i));
    }

    double r = out[0];
    delete [] out;

    return r;
}

template<>
auto h5_dataset::get<std::string>(const int i) {
    const hsize_t idx = (hsize_t)i;
    hsize_t dims = 1;
    hsize_t dim_sizes[] = {1};
    size_t  sdim;

    char *out;

    auto dset =  H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
    auto filetype = H5Dget_type(dset);

    // Initialize sdim with size of string + null terminator
    sdim = H5Tget_size(filetype) + 1;

    // Initialize output buffer
    out = (char *) malloc (sdim * sizeof (char));

    H5Dclose(dset);
    H5Tclose(filetype);

    // Open dataset and dataspace
    dset =  H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
    auto dspace = H5Dget_space(dset);

    // Select element to read
    H5Sselect_elements(dspace, H5S_SELECT_SET, 1, &idx);
    hid_t out_mem = H5Screate_simple(dims, dim_sizes, NULL);

    // Select right datatype
    auto memtype = H5Tcopy(H5T_C_S1);
    H5Tset_size (memtype, sdim);

    auto status = H5Dread(dset, memtype, out_mem, dspace, H5P_DEFAULT, out);

    if (status < 0) {
        throw std::runtime_error("error reading dataset " + name() + " at " + std::to_string(i));
    }

    std::string ret(out);

    free (out);
    H5Dclose(dset);
    H5Sclose(dspace);
    H5Tclose(memtype);

    return ret;
}

template<>
auto h5_dataset::get<std::vector<int>>(const int i, const int j) {
    hsize_t offset = i;
    hsize_t count = j-i;
    hsize_t stride = 1;
    hsize_t  block = 1;
    hsize_t dimsm = count;

    int rdata[count];

    auto id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
    hid_t dspace = H5Dget_space(id);

    hid_t out_mem = H5Screate_simple(1, &dimsm, NULL);

    H5Sselect_hyperslab(dspace, H5S_SELECT_SET, &offset, &stride, &count, &block);
    auto status = H5Dread(id, H5T_NATIVE_INT, out_mem, dspace, H5P_DEFAULT, rdata);

    H5Sclose(dspace);
    H5Sclose(out_mem);
    H5Dclose(id);

    if (status < 0) {
        throw std::runtime_error("error reading dataset " + name() + " from " + std::to_string(i) + " to " + std::to_string(j));
    }

    std::vector<int> out(rdata, rdata + count);

    return out;
}

template <>
auto h5_dataset::get<std::vector<double>>(const int i, const int j) {
    hsize_t offset = i;
    hsize_t count = j-i;
    hsize_t stride = 1;
    hsize_t  block = 1;
    hsize_t dimsm = count;

    double rdata[count];

    auto id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
    hid_t dspace = H5Dget_space(id);

    hid_t out_mem = H5Screate_simple(1, &dimsm, NULL);

    H5Sselect_hyperslab(dspace, H5S_SELECT_SET, &offset, &stride, &count, &block);

    auto status = H5Dread(id, H5T_NATIVE_DOUBLE, out_mem, dspace, H5P_DEFAULT, rdata);

    H5Sclose(dspace);
    H5Sclose(out_mem);
    H5Dclose(id);

    if (status < 0) {
        throw std::runtime_error("error reading dataset " + name() + " from " + std::to_string(i) + " to " + std::to_string(j));
    }

    std::vector<double> out(rdata, rdata + count);

    return out;
}

template <>
auto h5_dataset::get<std::pair<int,int>>(const int i) {
    const hsize_t idx_0[2] = {(hsize_t)i, (hsize_t)0};

    // Output
    int out_0, out_1;

    // Output dimensions 1x1
    hsize_t dims = 1;
    hsize_t dim_sizes[] = {1};

    // Output size
    hsize_t num_elements = 1;

    auto id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);
    hid_t dspace = H5Dget_space(id);

    H5Sselect_elements(dspace, H5S_SELECT_SET, num_elements, idx_0);
    hid_t out_mem_0 = H5Screate_simple(dims, dim_sizes, NULL);

    auto status0 = H5Dread(id, H5T_NATIVE_INT, out_mem_0, dspace, H5P_DEFAULT, &out_0);

    const hsize_t idx_1[2] = {(hsize_t)i, (hsize_t)1};

    H5Sselect_elements(dspace, H5S_SELECT_SET, num_elements, idx_1);
    hid_t out_mem_1 = H5Screate_simple(dims, dim_sizes, NULL);

    auto status1 = H5Dread(id, H5T_NATIVE_INT, out_mem_1, dspace, H5P_DEFAULT, &out_1);

    H5Sclose(dspace);
    H5Sclose(out_mem_0);
    H5Sclose(out_mem_1);
    H5Dclose(id);

    if (status0 < 0 || status1 < 0) {
        throw std::runtime_error("error reading dataset " + name() + " at " + std::to_string(i));
    }

    return std::make_pair(out_0, out_1);
}
template <>
auto h5_dataset::get<std::vector<int>>() {
    int out_a[size_];
    auto id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);

    auto status = H5Dread(id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            out_a);
    H5Dclose(id);

    if (status < 0) {
        throw std::runtime_error("error reading dataset " + name());
    }

    std::vector<int> out(out_a, out_a + size_);

    return out;
}

template <>
auto h5_dataset::get<std::vector<std::pair<int, int>>>() {
    int out_a[size_][2];
    auto id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);

    auto status = H5Dread(id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_a);

    H5Dclose(id);

    if (status < 0) {
        throw std::runtime_error("error reading dataset " + name());
    }

    std::vector<std::pair<int, int>> out(size_);
    for (unsigned i = 0; i < size_; i++) {
        out[i] = std::make_pair(out_a[i][0], out_a[i][1]);
    }

    return out;
}

template <>
auto h5_dataset::get<std::vector<std::pair<double, double>>>() {
    double out_a[size_][2];
    auto id = H5Dopen(parent_id_, name_.c_str(), H5P_DEFAULT);

    auto status = H5Dread(id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_a);

    H5Dclose(id);

    if (status < 0) {
        throw std::runtime_error("error reading dataset " + name());
    }

    std::vector<std::pair<double, double>> out(size_);
    for (unsigned i = 0; i < size_; i++) {
        out[i] = std::make_pair(out_a[i][0], out_a[i][1]);
    }

    return out;
}

///h5_group methods

h5_group::h5_group(hid_t parent, std::string name): parent_id_(parent), name_(name), group_h_(parent_id_, name_) {

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

std::shared_ptr<h5_group> h5_group::add_group(std::string name) {
    auto new_group = std::make_shared<h5_group>(group_h_.id, name);
    groups_.emplace_back(new_group);
    return new_group;
}

template <typename T>
void h5_group::add_dataset(std::string name, std::vector<T> dset) {
    auto new_dataset = std::make_shared<h5_dataset>(group_h_.id, name, dset);
    datasets_.emplace_back(new_dataset);
}

std::string h5_group::name() {
    return name_;
}

///h5_file methods
h5_file::h5_file(std::string name, bool new_file):
        name_(name),
        file_h_(name, new_file),
        top_group_(std::make_shared<h5_group>(file_h_.id, "/")) {}

std::string h5_file::name() {
    return name_;
}


///h5_wrapper methods

h5_wrapper::h5_wrapper() {}

h5_wrapper::h5_wrapper(const std::shared_ptr<h5_group>& g): ptr_(g) {
    unsigned i = 0;
    for (auto d: ptr_->datasets()) {
        dset_map_[d->name()] = i++;
    }
    i = 0;
    for (auto g: ptr_->groups()) {
        member_map_[g->name()] = i++;
        members_.emplace_back(g);
    }
}

int h5_wrapper::size() {
    return members_.size();
}

int h5_wrapper::find_group(std::string name) const {
    if (member_map_.find(name) != member_map_.end()) {
        return member_map_.at(name);
    }
    return -1;
}

int h5_wrapper::find_dataset(std::string name) const {
    if (dset_map_.find(name) != dset_map_.end()) {
        return dset_map_.at(name);
    }
    return -1;
}

int h5_wrapper::dataset_size(std::string name) const {
    if (dset_map_.find(name) != dset_map_.end()) {
        return ptr_->datasets().at(dset_map_.at(name))->size();
    }
    return -1;
}

template <typename T>
T h5_wrapper::get(std::string name, unsigned i) const {
    if (find_dataset(name) != -1) {
        return ptr_->datasets().at(dset_map_.at(name))->get<T>(i);
    }
    throw std::runtime_error("error reading dataset " + name);
}

template <typename T>
T h5_wrapper::get(std::string name, unsigned i, unsigned j) const {
    if (find_dataset(name)!= -1) {
        return ptr_->datasets().at(dset_map_.at(name))->get<T>(i, j);
    }
    throw std::runtime_error("error reading dataset " + name);
}

template <typename T>
T h5_wrapper::get(std::string name) const {
    if (find_dataset(name)!= -1) {
        return ptr_->datasets().at(dset_map_.at(name))->get<T>();
    }
    throw std::runtime_error("error reading dataset " + name);
}

const h5_wrapper& h5_wrapper::operator [](unsigned i) const {
    if (i < members_.size() && i >= 0) {
        return members_.at(i);
    }
    throw std::runtime_error("h5_wrapper index out of range " + name());
}

const h5_wrapper& h5_wrapper::operator [](std::string name) const {
    auto gid = find_group(name);
    if (gid != -1) {
        return members_.at(gid);
    }
    throw std::runtime_error("h5_wrapper index out of range " + name);
}

std::string h5_wrapper::name() const {
    return ptr_->name();
}

template void h5_group::add_dataset<int>(std::string, std::vector<int>);
template void h5_group::add_dataset<double>(std::string, std::vector<double>);
template void h5_group::add_dataset<std::vector<int>>(std::string, std::vector<std::vector<int>>);
template void h5_group::add_dataset<std::vector<double>>(std::string, std::vector<std::vector<double>>);

template int h5_wrapper::get<int>(std::string, unsigned) const;
template double h5_wrapper::get<double>(std::string, unsigned) const;
template std::string h5_wrapper::get<std::string>(std::string, unsigned) const;
template std::pair<int,int> h5_wrapper::get<std::pair<int,int>>(std::string, unsigned) const;

template std::vector<int> h5_wrapper::get<std::vector<int>>(std::string name, unsigned i, unsigned j) const;
template std::vector<double> h5_wrapper::get<std::vector<double>>(std::string name, unsigned i, unsigned j) const;

template std::vector<int> h5_wrapper::get<std::vector<int>>(std::string) const;
template std::vector<std::pair<int,int>> h5_wrapper::get<std::vector<std::pair<int,int>>>(std::string) const;
template std::vector<std::pair<double,double>> h5_wrapper::get<std::vector<std::pair<double,double>>>(std::string) const;
