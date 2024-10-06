#include "tau/Simplification/LSE.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirOps.h"
#include "tau/Core/ReachingDefs.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace tau;
using namespace tau::core;

namespace {

inline air::LoadOp getLoadForValue(mlir::Value Value) {
  return dyn_cast_or_null<air::LoadOp>(Value.getDefiningOp());
}

class LoadStoreEliminationPass final
    : public PassWrapper<LoadStoreEliminationPass, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "lse"; }
  StringRef getDescription() const override {
    return "Perform load-store elimination";
  }

  void runOnOperation() override {
    const auto &ReachingDefsAnalysis = getAnalysis<ReachingDefs>();
    OpBuilder Builder(&getContext());
    llvm::SmallDenseMap<air::LoadOp, mlir::Value> ReplaceWithInitial;

    // First, we walk over all load instructions and find the ones that we can
    // eliminate.
    getOperation().walk([&](air::LoadOp Load) {
      Value Address = Load.getAddress();
      const auto Definitions =
          ReachingDefsAnalysis.getDefinitions(*Load.getOperation(), Address);

      // We can eliminate a load if it has only one reaching definition.
      if (Definitions.size() == 1) {
        ReplaceWithInitial.insert({Load, Definitions.front()});
      }

      // TODO: support multiple possible loads by
    });

    // We can't just replace loads with definitions because definitions can also
    // be loads that we need to replace and so on.
    // For this reason, we make another map (it's actually a graph) where we
    // contract such chains of loads to be eliminated.
    llvm::SmallDenseMap<air::LoadOp, mlir::Value> ReplaceWithFinal;

    for (const auto [Load, Value] : ReplaceWithInitial) {
      // While processing other loads, we might've already processed this load
      // and found the exact value to replace it with.
      if (ReplaceWithFinal.contains(Load))
        continue;

      // The chaining described above is only possible if the value is a result
      // of the load as well.
      if (auto ValueAsLoad = getLoadForValue(Value)) {
        // We might've processed that load already, and if we have, let's use
        // that value for this load and stop.
        if (const auto FinalizedValueIt = ReplaceWithFinal.find(ValueAsLoad);
            FinalizedValueIt != ReplaceWithFinal.end()) {
          ReplaceWithFinal.insert({Load, FinalizedValueIt->getSecond()});
          continue;
        }

        // Otherwise, let's see find the best value we can replace this load
        // with.
        mlir::Value ValueToReplaceWith{};
        // This collection will contain all of the loads we want to replace with
        // the same value.
        SmallVector<air::LoadOp, 4> LoadsSharingValue{Load};

        // While we want to replace a load with a value of another load, we keep
        // going.
        for (air::LoadOp CurrentLoad = ValueAsLoad; CurrentLoad;
             // If we execute at least one iteration and get here,
             // we have definitely initialized ValueToReplaceWith.
             CurrentLoad = getLoadForValue(ValueToReplaceWith)) {
          if (const auto IntermediateValueToReplaceWithIt =
                  ReplaceWithInitial.find(CurrentLoad);
              IntermediateValueToReplaceWithIt != ReplaceWithInitial.end()) {
            ValueToReplaceWith = IntermediateValueToReplaceWithIt->getSecond();
            LoadsSharingValue.push_back(CurrentLoad);
          } else {
            // We only continue if that load is also lined up for elimination.
            break;
          }
        }

        if (ValueToReplaceWith) {
          for (auto LoadToReplace : LoadsSharingValue) {
            ReplaceWithFinal.insert({LoadToReplace, ValueToReplaceWith});
          }
          break;
        }
      }

      // If we got here, there is nothing special about this load-value pair,
      // and we should add it to the final map as-is.
      ReplaceWithFinal.insert({Load, Value});
    }

    // Now let's eliminate loads without the risk of deleting something
    // that we might depend on later on.
    for (auto [Load, Value] : ReplaceWithFinal) {
      Builder.setInsertionPointAfterValue(Load);
      mlir::Value NoOp = Builder.create<air::NoOp>(Load.getLoc(), Value);
      Load.replaceAllUsesWith(NoOp);
      Load->erase();
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoadStoreEliminationPass);
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> tau::simple::createLSEPass() {
  return std::make_unique<LoadStoreEliminationPass>();
}
