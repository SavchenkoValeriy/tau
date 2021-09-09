#include "tau/AIR/AirOps.h"
#include "tau/Checkers/Checkers.h"
#include "tau/Core/State.h"

#include <mlir/IR/BuiltinAttributes.h>
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

// TODO: Add checkers API
class UseOfUninit : public chx::Checker<UseOfUninit> {
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
      // TODO: Think about a better marking mechanism
      mark(Address, UNINIT);
      auto Fused = Address->getLoc().dyn_cast<FusedLoc>();
      llvm::SourceMgr SourceMgr;
      SourceMgrDiagnosticHandler Handler(SourceMgr, Store.getContext(),
                                         llvm::errs());
      Address->getOpOperand(10);

      Address->emitError() << "Use of uninitialized value";
    });
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> tau::chx::createUseOfUninitChecker() {
  return std::make_unique<UseOfUninit>();
}

void tau::chx::registerUseOfUninitChecker() { PassRegistration<UseOfUninit>(); }
