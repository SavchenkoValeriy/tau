#include "tau/AIR/AirOps.h"
#include "tau/Checkers/Checkers.h"

#include <memory>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

using namespace tau;
using namespace air;
using namespace mlir;

namespace {

// TODO: Add checkers API
class UseOfUninit : public PassWrapper<UseOfUninit, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "use-of-uninit"; }
  StringRef getDescription() const override {
    return "Detect uses of uninitialized values";
  }

  void runOnOperation() override {
    FuncOp F = getOperation();

    F.walk([](StoreOp Store) {
      auto StoredValueSource = Store.getValue().getDefiningOp<UndefOp>();
      if (!StoredValueSource)
        return;

      mlir::Operation *Address = Store.getAddress().getDefiningOp();
      // TODO: Think about a better marking mechanism
      Address->setAttr("use-of-uninit",
                       BoolAttr::get(Store.getContext(), true));
    });
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> tau::chx::createUseOfUninitChecker() {
  return std::make_unique<UseOfUninit>();
}

void tau::chx::registerUseOfUninitChecker() { PassRegistration<UseOfUninit>(); }
