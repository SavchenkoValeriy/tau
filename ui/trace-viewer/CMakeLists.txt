# Build trace viewer
add_custom_target(build-trace-viewer
  COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_SOURCE_DIR}/ui/trace-viewer
      ${CMAKE_BINARY_DIR}/ui/trace-viewer
  COMMAND ${BUN_EXECUTABLE} install
  COMMAND ${BUN_EXECUTABLE} run build
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ui/trace-viewer
  DEPENDS ${BUN_EXECUTABLE}
  COMMENT "Building trace viewer"
)

# Serve trace viewer
add_custom_target(trace-viewer
  DEPENDS build-trace-viewer
  COMMAND ${BUN_EXECUTABLE} run serve
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ui/trace-viewer
  COMMENT "Serving trace viewer"
)
