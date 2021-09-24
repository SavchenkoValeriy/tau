add_tau_unittest(CoreTests
  FlowSensitiveTest.cpp)

target_link_libraries(CoreTests
  PRIVATE
  tauCore
  )
