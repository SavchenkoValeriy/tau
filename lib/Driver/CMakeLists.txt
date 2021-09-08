add_tau_executable(tau-cc tau-cc.cpp)

target_link_libraries(tau-cc PRIVATE
  tauFrontendClang
  tauCheckers
  clangTooling
  MLIRPass)

install(TARGETS tau-cc
  RUNTIME DESTINATION bin)
