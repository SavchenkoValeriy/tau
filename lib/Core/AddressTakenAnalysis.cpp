#include "tau/Core/AddressTakenAnalysis.h"
#include "tau/AIR/AirOps.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace tau;

tau::core::AddressTakenAnalysis::AddressTakenAnalysis(Operation *Op) {
  assert(isa<FuncOp>(Op) &&
         "Address-taken analysis is only available for functions");

  FuncOp Function = cast<FuncOp>(Op);
  // TODO: support heap allocations
  Function.walk([this](air::AllocaOp Alloca) {
    for (auto &Use : Alloca.getResult().getUses()) {
      if (isa<air::LoadOp>(Use.getOwner()) ||
          isa<air::DeallocaOp>(Use.getOwner()) ||
          (isa<air::StoreOp>(Use.getOwner()) && Use.getOperandNumber() != 0))
        continue;

      AddressTakenValues.insert(Alloca);
    }
  });
}

bool tau::core::AddressTakenAnalysis::hasAddressBeenTaken(
    mlir::Value Candidate) const {
  return isa<air::AllocaOp>(Candidate.getDefiningOp()) &&
         AddressTakenValues.contains(Candidate);
}
