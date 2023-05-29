#include "tau/Core/EscapeAnalysis.h"
#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirTypes.h"
#include "tau/Core/AliasAnalysis.h"
#include "tau/Core/PointsToAnalysis.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/AnalysisManager.h>
#include <queue>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace tau;

namespace tau::core {

EscapeAnalysis::EscapeAnalysis(Operation *FunctionOp, AnalysisManager &AM) {
  assert(isa<FuncOp>(FunctionOp) &&
         "Address-taken analysis is only available for functions");

  FuncOp Function = cast<FuncOp>(FunctionOp);
  const auto &PointsTo = AM.getAnalysis<PointsToAnalysis>();
  const auto &Aliases = AM.getAnalysis<AliasAnalysis>();

  Function.walk([&](CallOp Call) {
    std::queue<Value> CallEscapes;
    for (Value Arg : Call.getArgOperands()) {
      if (Arg.getType().isa<air::PointerType>()) {
        CallEscapes.push(Arg);
        for (Value ArgAlias : Aliases.getAliases(Arg))
          CallEscapes.push(ArgAlias);
      }
    }
    while (!CallEscapes.empty()) {
      Value EscapedValue = CallEscapes.front();
      CallEscapes.pop();

      Type PointeeType =
          EscapedValue.getType().cast<air::PointerType>().getElementType();
      if (!Escapes.insert(EscapedValue).second ||
          !PointeeType.isa<air::PointerType>())
        continue;

      for (Value Pointee : PointsTo.getPointsToSet(EscapedValue))
        CallEscapes.push(Pointee);
    }
  });
}

} // end namespace tau::core
