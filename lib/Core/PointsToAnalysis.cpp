#include "tau/Core/PointsToAnalysis.h"
#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirTypes.h"
#include "tau/Core/AddressTakenAnalysis.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/AnalysisManager.h>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace tau;

namespace tau::core {
PointsToAnalysis::PointsToSet PointsToAnalysis::Empty = {};

PointsToAnalysis::PointsToAnalysis(Operation *Op, AnalysisManager &AM) {
  assert(isa<FuncOp>(Op) &&
         "Address-taken analysis is only available for functions");

  FuncOp Function = cast<FuncOp>(Op);
  const auto &AddressTaken = AM.getAnalysis<AddressTakenAnalysis>();

  llvm::DenseMap<Type, llvm::SmallVector<Value, 4>> PointersByType;
  Function.walk([this, &PointersByType](air::AllocaOp Alloca) {
    // All allocas should have pointer types by construction
    const auto &AllocaType = Alloca.getType().cast<air::PointerType>();
    if (AllocaType.getElementType().isa<air::PointerType>()) {
      PointersByType[AllocaType.getElementType()].push_back(Alloca);
    }
  });

  for (Value Escaped : AddressTaken.getAddressTakenValues()) {
    const auto It = PointersByType.find(Escaped.getType());
    if (It == PointersByType.end())
      continue;
    for (Value Pointer : It->getSecond()) {
      Sets[Pointer].push_back(Escaped);
    }
  }
}
} // end namespace tau::core
