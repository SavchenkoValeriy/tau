#include "tau/Checkers/Registry.h"
#include "tau/Checkers/Checkers.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ManagedStatic.h>
#include <memory>
#include <mlir/Support/TypeID.h>

using namespace tau::chx;
using namespace mlir;
using namespace llvm;

static ManagedStatic<StringMap<std::unique_ptr<Checker>>> CheckerRegistry;
static ManagedStatic<StringMap<TypeID>> CheckerIDRegistry;

void registerChecker(const CheckerAllocatorFunction &Constructor) {
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
