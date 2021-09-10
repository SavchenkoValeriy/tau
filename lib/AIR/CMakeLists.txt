add_mlir_dialect_library(tauAir
  AirDialect.cpp
  AirOps.cpp
  AirAttrs.cpp
  AirTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Air

  DEPENDS
  MLIRAirOpsIncGen

  LINK_LIBS
  PRIVATE
  MLIRIR
  )
