#include "tau/Checkers/Registry.h"
#include "tau/Checkers/Checkers.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ManagedStatic.h>
#include <memory>
#include <mlir/Support/TypeID.h>

using namespace tau::chx;
using namespace mlir;
using namespace llvm;

//===----------------------------------------------------------------------===//
//                             Checker registration
//===----------------------------------------------------------------------===//

static ManagedStatic<StringMap<std::unique_ptr<Checker>>> CheckerRegistry;
static ManagedStatic<StringMap<TypeID>> CheckerIDRegistry;

void tau::chx::registerChecker(const CheckerAllocatorFunction &Constructor) {
  llvm::errs() << "HELLO\n";
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

private:
  StringMap<cl::opt<bool>> EnabledCheckers;
};

CheckerCLParser::CheckerCLParser(cl::OptionCategory &CheckersCategory)
    : PImpl(new CheckerCLParser::Implementation(CheckersCategory)) {}

CheckerCLParser::CheckerCLParser(CheckerCLParser &&) = default;
CheckerCLParser &CheckerCLParser::operator=(CheckerCLParser &&) = default;

CheckerCLParser::~CheckerCLParser() = default;
