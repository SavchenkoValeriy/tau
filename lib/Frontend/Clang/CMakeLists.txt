add_tau_library(
  tauFrontendClang
  AIRGenAction.cpp
  AIRGenerator.cpp
  Clang.cpp
)
target_link_libraries(
  tauFrontendClang
  PRIVATE
  tauAir
  immer
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
)
