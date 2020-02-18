add_executable(net EXCLUDE_FROM_ALL network.cpp common/cpp/hdf5_lib.cpp)
add_dependencies(examples net)

find_package(HDF5 REQUIRED)

target_link_libraries(net PRIVATE arbor arborenv arbor-sup ext-tclap ext-json ${HDF5_C_LIBRARIES})
target_include_directories(net PRIVATE common/cpp/include ${HDF5_INCLUDE_DIRS})
