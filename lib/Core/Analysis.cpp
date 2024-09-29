#include "tau/Core/Analysis.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerRegistry.h"
#include "tau/Core/Events.h"
#include "tau/Core/FlowSensitive.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

using namespace tau;
using namespace tau::core;
using namespace mlir;
using namespace mlir::func;

namespace {

class MainAnalysis final
    : public PassWrapper<MainAnalysis, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "main-analysis-pass"; }
  StringRef getDescription() const override {
    return "Perform analysis to find issues for all checkers";
  }

  void runOnOperation() override {
    if (getOperation().isDeclaration())
      return;

    auto &FlowSen = getAnalysis<FlowSensitiveAnalysis>();

    // TODO: the last stage should be path-sensitive analysis.
    for (auto &FoundIssue : FlowSen.getFoundIssues()) {
      // For every found issue, we need to find the corresponding
      // checker and string the events through it to produce
      // the actual error visible to the user.
      const StateEvent &ErrorEvent = FoundIssue.ErrorEvent;
      auto &Checker = findChecker(ErrorEvent.getKey().CheckerID);

      InFlightDiagnostic Error = Checker.emitError(ErrorEvent.getLocation(),
                                                   ErrorEvent.getKey().State);

      // TODO: remove and replace with proper parent event traversal
      const auto GetParent = [](const StateEvent &E) -> const StateEvent * {
        const auto Parents = E.getParents();
        if (Parents.empty())
          return nullptr;

        return Parents.front().get<const StateEvent *>();
      };
      // TODO: sort parents and traverse bottom up.

      // Traverse the event tree and emit additional notes.
      for (const StateEvent *CurrentEvent = GetParent(ErrorEvent); CurrentEvent;
           CurrentEvent = GetParent(*CurrentEvent))
        Checker.emitNote(Error, CurrentEvent->getLocation(),
                         CurrentEvent->getKey().State);
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MainAnalysis);
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> tau::core::createMainAnalysis() {
  return std::make_unique<MainAnalysis>();
}
