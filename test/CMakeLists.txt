configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  )

list(APPEND TAU_TEST_DEPS
  tau-cc
  unittests
  )

add_lit_testsuite(check-cc "Running regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${TAU_TEST_PARAMS}
  DEPENDS ${TAU_TEST_DEPS}
  ARGS ${TAU_TEST_EXTRA_ARGS}
  )
set_target_properties(check-cc PROPERTIES FOLDER "Tau tests")
add_dependencies(check check-cc)
