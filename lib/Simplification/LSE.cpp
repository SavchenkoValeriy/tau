#include "tau/Simplification/LSE.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirOps.h"
#include "tau/Core/ReachingDefs.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace tau;
using namespace tau::core;

namespace {

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

    getOperation().walk([&](air::LoadOp Load) {
      Value Address = Load.getAddress();
      const auto Definitions =
          ReachingDefsAnalysis.getDefinitions(*Load.getOperation(), Address);

      if (Definitions.size() == 1) {
        Builder.setInsertionPointAfterValue(Load);
        mlir::Value NoOp =
            Builder.create<air::NoOp>(Load.getLoc(), Definitions.front());
        Load.replaceAllUsesWith(NoOp);
        Load->erase();
      }

      // TODO: support multiple possible loads by
    });
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoadStoreEliminationPass);
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> tau::simple::createLSEPass() {
  return std::make_unique<LoadStoreEliminationPass>();
}
