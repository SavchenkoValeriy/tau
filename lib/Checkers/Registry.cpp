#include "tau/Checkers/Registry.h"
#include "tau/AIR/AirAttrs.h"
#include "tau/Core/Checker.h"

#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ManagedStatic.h>
#include <memory>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/TypeID.h>

using namespace tau;
using namespace chx;
using namespace core;
using namespace mlir;
using namespace llvm;

//===----------------------------------------------------------------------===//
//                             Checker registration
//===----------------------------------------------------------------------===//

static ManagedStatic<StringMap<std::unique_ptr<Checker>>> CheckerRegistry;
static ManagedStatic<StringMap<TypeID>> CheckerIDRegistry;

void tau::chx::registerChecker(const CheckerAllocatorFunction &Constructor) {
  std::unique_ptr<Checker> NewChecker = Constructor();
  StringRef Argument = NewChecker->getArgument();
  if (Argument.empty())
    report_fatal_error(
        "Trying to register a pass that does not override `getArgument()`: " +
        NewChecker->getName());

  TypeID CheckerID = NewChecker->getTypeID();

  CheckerRegistry->try_emplace(Argument, std::move(NewChecker));
  if (auto It = CheckerIDRegistry->try_emplace(Argument, CheckerID).first;
      It->second != CheckerID) {
    report_fatal_error(
        "Trying to register a different checker for the same argument '" +
        Argument + "'");
  }
}

//===----------------------------------------------------------------------===//
//                            Pass running checkers
//===----------------------------------------------------------------------===//

namespace {

air::StateChangeAttr getStateChangeAttr(Operation *Op, StringRef CheckerID) {
  if (auto StateAttributes = Op->getAttrOfType<ArrayAttr>(StateAttrID)) {
    const auto *It =
        llvm::find_if(StateAttributes, [CheckerID](Attribute Attr) {
          if (auto StateChange = Attr.dyn_cast<air::StateChangeAttr>())
            return StateChange.getCheckerID() == CheckerID;
          return false;
        });
    return It != StateAttributes.end() ? It->cast<air::StateChangeAttr>()
                                       : air::StateChangeAttr{};
  }

  return {};
}

class EnabledCheckersPass final
    : public PassWrapper<EnabledCheckersPass, OperationPass<FuncOp>> {
public:
  using Checkers = llvm::SmallVector<Checker *, 32>;

  EnabledCheckersPass(Checkers &&EnabledCheckers)
      : EnabledCheckers(std::move(EnabledCheckers)) {}

  StringRef getArgument() const override { return "run-enabled-checkers"; }
  StringRef getDescription() const override {
    return "Run all enabled checkers on the given function";
  }
  void runOnOperation() override {
    FuncOp Function = getOperation();

    Function.walk([this](Operation *Op) {
      for (Checker *EnabledChecker : EnabledCheckers) {
        EnabledChecker->process(Op);

        // FIXME: This is only an example of how we can report errors
        //        without any particular information about checkers.
        //        It should be removed and replaced with a separate pass
        //        that actually performs the analysis.
        if (auto ErrorAttr =
                getStateChangeAttr(Op, EnabledChecker->getArgument())) {
          air::StateID ID = ErrorAttr.getToState();
          if (!ID.isError())
            continue;
          mlir::Operation *OperandOp =
              Op->getOperand(ErrorAttr.getOperandIdx()).getDefiningOp();

          if (auto ChangeAttr = getStateChangeAttr(
                  OperandOp, EnabledChecker->getArgument())) {
            if (ErrorAttr.getFromState() != ChangeAttr.getToState())
              continue;

            auto Diag = EnabledChecker->emitError(Op, ID);
            EnabledChecker->emitNote(Diag, OperandOp, ChangeAttr.getToState());
          }
        }
      }
    });
  }

private:
  Checkers EnabledCheckers;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                               CheckerCLParser
//===----------------------------------------------------------------------===//

class CheckerCLParser::Implementation {
public:
  Implementation(cl::OptionCategory &CheckersCategory) {
    for (auto &Entry : *CheckerRegistry) {
      EnabledCheckers.try_emplace(Entry.getKey(), Entry.getKey(),
                                  cl::desc(Entry.getValue()->getDescription()),
                                  cl::cat(CheckersCategory));
    }
  }

  void addEnabledCheckers(PassManager &PM) {
    EnabledCheckersPass::Checkers ListOfEnabledCheckers;
    // First get a subset of checkers enabled by options.
    auto FilteredCheckers =
        llvm::make_filter_range(*CheckerRegistry, [this](auto &Entry) -> bool {
          auto It = EnabledCheckers.find(Entry.getKey());
          assert(It != EnabledCheckers.end() && "Encountered unknown checker");
          return It->getValue();
        });

    // And then convert those into a plain vector of checker pointers.
    llvm::transform(FilteredCheckers, std::back_inserter(ListOfEnabledCheckers),
                    [](auto &Entry) { return Entry.getValue().get(); });

    // We evaluate all checkers in one pass.
    PM.addNestedPass<FuncOp>(std::make_unique<EnabledCheckersPass>(
        std::move(ListOfEnabledCheckers)));
  }

private:
  StringMap<cl::opt<bool>> EnabledCheckers;
};

CheckerCLParser::CheckerCLParser(cl::OptionCategory &CheckersCategory)
    : PImpl(new CheckerCLParser::Implementation(CheckersCategory)) {}

CheckerCLParser::CheckerCLParser(CheckerCLParser &&) = default;
CheckerCLParser &CheckerCLParser::operator=(CheckerCLParser &&) = default;

CheckerCLParser::~CheckerCLParser() = default;

void CheckerCLParser::addEnabledCheckers(mlir::PassManager &PM) {
  PImpl->addEnabledCheckers(PM);
}
