#include "tau/Core/CheckerPass.h"

#include "tau/Core/Checker.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <memory>

using namespace tau::core;
using namespace mlir;
using namespace mlir::func;
using namespace llvm;

class CheckersPass final
    : public PassWrapper<CheckersPass, OperationPass<FuncOp>> {
public:
  using Checkers = SmallVector<AbstractChecker *, 32>;

  CheckersPass(ArrayRef<AbstractChecker *> CheckersToRun)
      : EnabledCheckers(CheckersToRun.begin(), CheckersToRun.end()) {}

  StringRef getArgument() const override { return "run-enabled-checkers"; }
  StringRef getDescription() const override {
    return "Run all enabled checkers on the given function";
  }
  void runOnOperation() override {
    FuncOp Function = getOperation();

    Function.walk([this](Operation *Op) {
      for (AbstractChecker *EnabledChecker : EnabledCheckers) {
        EnabledChecker->process(Op);
      }
    });
  }

private:
  Checkers EnabledCheckers;
};

std::unique_ptr<Pass>
tau::core::createCheckerPass(ArrayRef<AbstractChecker *> CheckersToRun) {
  return std::make_unique<CheckersPass>(CheckersToRun);
}
