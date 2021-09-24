add_custom_target(unittests)
add_dependencies(check unittests)
add_custom_command(
  TARGET unittests
  COMMENT "Run unit-tests"
  POST_BUILD
  COMMAND ctest --output-on-failure)

add_subdirectory(Core)
