add_tau_unittest(CoreTests
  AddressTakenAnalysisTest.cpp
  CheckerTest.cpp
  FlowSensitiveTest.cpp
  PointsToAnalysisTest.cpp
  ReachingDefsTest.cpp
  TopoOrderEnumeratorTest.cpp
  )

target_link_libraries(CoreTests
  PRIVATE
  tauCore
  tauFrontendClang
  clangTooling
  MLIRPass
  )
