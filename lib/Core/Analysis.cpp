#include "tau/Core/Analysis.h"
#include "tau/Core/FlowSensitive.h"
#include "tau/Core/FlowWorklist.h"

#include <mlir/Pass/Pass.h>

#include <memory>

using namespace tau;
using namespace tau::core;
using namespace mlir;

namespace {

class MainAnalysis final
    : public PassWrapper<MainAnalysis, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "main-analysis-pass"; }
  StringRef getDescription() const override {
    return "Perform analysis to find issues for all checkers";
  }

  void runOnOperation() override {
    auto Analysis = getAnalysis<ForwardWorklist>();
    FuncOp Function = getOperation();
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> tau::core::createMainAnalysis() {
  return std::make_unique<MainAnalysis>();
}
