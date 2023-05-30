add_tau_library(tauSimplification
  LSE.cpp
  Simplification.cpp

  LINK_LIBS
  tauCore
  tauAir

  DEPENDS
  tauAir
)
