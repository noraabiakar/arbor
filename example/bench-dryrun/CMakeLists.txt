add_executable(bench-dryrun bench-dryrun.cpp recipe.cpp parameters.cpp)

target_link_libraries(bench-dryrun PRIVATE arbor arbor-aux ext-json)
