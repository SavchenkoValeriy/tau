#include "tau/Core/Analysis.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerRegistry.h"
#include "tau/Core/FlowSensitive.h"
#include "tau/Core/StateEventForest.h"

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
    auto &FlowSen = getAnalysis<FlowSensitiveAnalysis>();

    // TODO: the last stage should be path-sensitive analysis.
    for (auto &FoundIssue : FlowSen.getFoundIssues()) {
      // For every found issue, we need to find the corresponding
      // checker and string the events through it to produce
      // the actual error visible to the user.
      const StateEvent &ErrorEvent = FoundIssue.ErrorEvent;
      auto &Checker = findChecker(ErrorEvent.Key.CheckerID);

      InFlightDiagnostic Error =
          Checker.emitError(ErrorEvent.Location, ErrorEvent.Key.State);

      // Traverse the event tree and emit additional notes.
      for (const StateEvent *CurrentEvent = ErrorEvent.Parent; CurrentEvent;
           CurrentEvent = CurrentEvent->Parent)
        Checker.emitNote(Error, CurrentEvent->Location,
                         CurrentEvent->Key.State);
    }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> tau::core::createMainAnalysis() {
  return std::make_unique<MainAnalysis>();
}
