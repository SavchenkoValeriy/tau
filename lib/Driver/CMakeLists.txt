add_tau_executable(tau-cc tau-cc.cpp)

target_link_libraries(tau-cc PRIVATE
  tauFrontendClang
  clangTooling
  MLIRIR)

install(TARGETS tau-cc
  RUNTIME DESTINATION bin)
