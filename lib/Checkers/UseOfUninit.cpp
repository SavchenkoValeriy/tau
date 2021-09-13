#include "tau/AIR/AirOps.h"
#include "tau/Checkers/Checkers.h"
#include "tau/Core/State.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

#include <memory>

using namespace tau;
using namespace air;
using namespace mlir;

using UninitState = core::State<1, 1>;
constexpr UninitState UNINIT = UninitState::getNonErrorState(0);
constexpr UninitState ERROR = UninitState::getErrorState(0);

namespace {

class UseOfUninit : public chx::Checker<UseOfUninit, UninitState> {
public:
  StringRef getArgument() const override { return "use-of-uninit"; }
  StringRef getDescription() const override {
    return "Detect uses of uninitialized values";
  }

  void runOnOperation() override {
    FuncOp F = getOperation();

    F.walk([this](StoreOp Store) {
      auto StoredValueSource = Store.getValue().getDefiningOp<UndefOp>();
      if (!StoredValueSource)
        return;

      mlir::Operation *Address = Store.getAddress().getDefiningOp();
      markResultChange(Address, UNINIT);
      Address->emitError() << "Use of uninitialized value";
    });

    F.walk([this](LoadOp Load) {
      markChange(Load.getOperation(), Load.getAddress(), UNINIT, ERROR);
      Load.emitError() << "Use of uninitialized value";
    });
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> tau::chx::createUseOfUninitChecker() {
  return std::make_unique<UseOfUninit>();
}

void tau::chx::registerUseOfUninitChecker() { PassRegistration<UseOfUninit>(); }
