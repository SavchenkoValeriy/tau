add_tau_unittest(CoreTests
  CheckerTest.cpp
  FlowSensitiveTest.cpp
  TopoOrderEnumeratorTest.cpp
  )

target_link_libraries(CoreTests
  PRIVATE
  tauCore
  tauFrontendClang
  clangTooling
  MLIRPass
  )
