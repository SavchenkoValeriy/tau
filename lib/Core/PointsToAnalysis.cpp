#include "tau/Core/PointsToAnalysis.h"
#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirTypes.h"
#include "tau/Core/AddressTakenAnalysis.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/AnalysisManager.h>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace tau;

namespace tau::core {
PointsToAnalysis::PointsToSet PointsToAnalysis::Empty = {};

PointsToAnalysis::PointsToAnalysis(Operation *FunctionOp, AnalysisManager &AM) {
  assert(isa<FuncOp>(FunctionOp) &&
         "Address-taken analysis is only available for functions");

  FuncOp Function = cast<FuncOp>(FunctionOp);
  const auto &AddressTaken = AM.getAnalysis<AddressTakenAnalysis>();

  llvm::DenseMap<Type, llvm::SmallVector<Value, 4>> PointersByType;
  const auto AddPointerByType = [&PointersByType](Value Pointer, Type T) {
    if (const auto PointerType = T.dyn_cast<air::PointerType>())
      if (PointerType.getElementType().isa<air::PointerType>())
        PointersByType[PointerType.getElementType()].push_back(Pointer);
  };

  Function.walk([this, &AddPointerByType](Operation *Op) {
    llvm::TypeSwitch<Operation *, void>(Op)
        .Case([&AddPointerByType](air::AllocaOp Alloca) {
          AddPointerByType(Alloca, Alloca.getType());
        })
        .Case([&AddPointerByType](air::LoadOp Load) {
          AddPointerByType(Load, Load.getType());
        })
        .Case([&AddPointerByType](CallOp Call) {
          if (Call.getResults().size() == 1)
            AddPointerByType(Call.getResults().front(),
                             Call.getResultTypes().front());
        });
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
