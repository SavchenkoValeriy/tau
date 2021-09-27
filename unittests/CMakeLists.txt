add_custom_target(unittests)
add_dependencies(check unittests)
add_custom_command(
  COMMAND ${CMAKE_COMMAND} -E echo "Running unit-tests"
  COMMAND ctest --output-on-failure
  VERBATIM
  PRE_BUILD
  TARGET unittests
  )

add_subdirectory(Core)
