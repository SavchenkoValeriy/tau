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

class UseOfUninit
    : public chx::CheckerWrapper<UseOfUninit, UninitState, StoreOp, LoadOp> {
public:
  StringRef getName() const override {
    return "Use of uninitialized value checker";
  }
  StringRef getArgument() const override { return "use-of-uninit"; }
  StringRef getDescription() const override {
    return "Detect uses of uninitialized values";
  }

  void process(StoreOp Store) const {
    auto StoredValueSource = Store.getValue().getDefiningOp<UndefOp>();
    if (!StoredValueSource)
      return;

    mlir::Operation *Address = Store.getAddress().getDefiningOp();
    markResultChange(Address, UNINIT);
  }

  void process(LoadOp Load) const {
    markChange(Load.getOperation(), Load.getAddress(), UNINIT, ERROR);
  }

  InFlightDiagnostic emitError(mlir::Operation *Op, UninitState State) {
    assert(State == ERROR);
    return Op->emitError("Use of uninitialized value");
  }

  void emitNote(InFlightDiagnostic &Diag, mlir::Operation *Op,
                UninitState State) {
    assert(State == UNINIT);
    Diag.attachNote(Op->getLoc()) << "Declared without initial value here";
  }
};

} // end anonymous namespace

