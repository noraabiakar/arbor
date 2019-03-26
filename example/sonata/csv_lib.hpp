#include <arbor/common_types.hpp>
#include <arbor/util/optional.hpp>
#include <string.h>
#include <stdio.h>
#include <hdf5.h>
#include <assert.h>

using arb::cell_size_type;

class csv_file {
    std::string fileName;
    char delimeter;

public:
    csv_file(std::string filename, char delm = ',') :
            fileName(filename), delimeter(delm) { }

    std::vector<std::vector<std::string>> get_data() {
        std::ifstream file(fileName);

        std::vector<std::vector<std::string>> data;
        std::string line;

        while (getline(file, line)) {
            std::vector<std::string> vec;
            std::stringstream ss(line);
            std::string value;

            while(getline(ss, value, delimeter)) {
                vec.push_back(value);
            }
            data.push_back(vec);
        }
        file.close();
        return data;
    }
};

////////////////////////////////////////////////////////

class csv_record {
public:
    csv_record(csv_file f) {
        auto csv_data = f.get_data();

        unsigned count = 0;
        auto vec = csv_data.front();
        for (auto field: vec) {
            field_map_[field] = count++;
        }

        data_.resize(count);
        for(auto it = csv_data.begin()+1; it < csv_data.end(); it++) {
            unsigned loc = 0;
            for (auto field: *it) {
                data_[loc++].push_back(field);
            };
        }
    }

    std::unordered_map<std::string, unsigned> map() {
        return field_map_;
    }

    std::vector<std::vector<std::string>> data() {
        return data_;
    }

private:
    std::unordered_map<std::string, unsigned> field_map_;
    std::vector<std::vector<std::string>> data_;
};
