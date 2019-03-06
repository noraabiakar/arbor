add_executable(sonata EXCLUDE_FROM_ALL sonata.cpp hdf5_lib.hpp)
add_dependencies(examples sonata)

find_package(HDF5)
if(HDF5_FOUND)
    target_link_libraries(arbor-private-deps INTERFACE ${HDF5_C_LIBRARIES})
    list(APPEND arbor_export_dependencies "HDF5")
endif()

target_link_libraries(sonata PRIVATE arbor arborenv arbor-sup ${HDF5_C_LIBRARIES})
