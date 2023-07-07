#include "tau/Simplification/Simplification.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirOps.h"
#include "tau/Simplification/LSE.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace tau;

namespace {
class SimplificationPass final
    : public PassWrapper<SimplificationPass, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "simplification"; }
  StringRef getDescription() const override {
    return "Perform all simplifications";
  }

  void runOnOperation() override {
    PassManager PM(&getContext(), OpPassManager::Nesting::Explicit,
                   FuncOp::getOperationName());
    PM.addPass(simple::createLSEPass());

    if (failed(PM.run(getOperation())))
      signalPassFailure();
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplificationPass);
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> tau::simple::createSimplification() {
  return std::make_unique<SimplificationPass>();
}
