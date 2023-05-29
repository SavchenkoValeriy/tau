#include "tau/Core/AliasAnalysis.h"
#include "tau/AIR/AirOps.h"
#include "tau/Core/PointsToAnalysis.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/AnalysisManager.h>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace tau;

namespace tau::core {
AliasAnalysis::Aliases AliasAnalysis::Empty = {};

AliasAnalysis::AliasAnalysis(Operation *Op, AnalysisManager &AM) {
  assert(isa<FuncOp>(Op) &&
         "Address-taken analysis is only available for functions");

  FuncOp Function = cast<FuncOp>(Op);
  const auto &PointsTo = AM.getAnalysis<PointsToAnalysis>();

  for (auto &[Pointer, Values] : PointsTo) {
    for (auto &Use : Pointer.getUses()) {
      if (auto Load = dyn_cast<air::LoadOp>(Use.getOwner())) {
        mlir::Value A = Load.getResult();
        for (auto B : Values) {
          Sets[A].insert(B);
          Sets[B].insert(A);
        }
      }
    }
  }
}
} // end namespace tau::core
