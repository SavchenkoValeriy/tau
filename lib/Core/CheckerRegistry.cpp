#include "tau/Core/CheckerRegistry.h"

#include "tau/AIR/AirAttrs.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerPass.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ManagedStatic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/TypeID.h>

#include <iterator>
#include <memory>

using namespace tau;
using namespace core;
using namespace mlir;
using namespace mlir::func;
using namespace llvm;

//===----------------------------------------------------------------------===//
//                             Checker registration
//===----------------------------------------------------------------------===//

static ManagedStatic<StringMap<std::unique_ptr<AbstractChecker>>>
    CheckerRegistry;
static ManagedStatic<StringMap<TypeID>> CheckerIDRegistry;

void tau::core::registerChecker(const CheckerAllocatorFunction &Constructor) {
  std::unique_ptr<AbstractChecker> NewChecker = Constructor();
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
    SmallVector<AbstractChecker *, 20> ListOfEnabledCheckers;
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
    PM.addNestedPass<FuncOp>(createCheckerPass(ListOfEnabledCheckers));
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

//===----------------------------------------------------------------------===//
//                              Find checker by ID
//===----------------------------------------------------------------------===//

AbstractChecker &tau::core::findChecker(llvm::StringRef ID) {
  auto It = CheckerRegistry->find(ID);
  assert(It != CheckerRegistry->end() &&
         "Got request about a non-registered checker");
  return *It->getValue().get();
}
