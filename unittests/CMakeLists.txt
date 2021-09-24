add_custom_target(unittests)
add_dependencies(check unittests)
add_dependencies(unittests test)

add_subdirectory(Core)
