#include "tau/Frontend/Clang/AIRGenAction.h"
#include "tau/Frontend/Clang/AIRGenerator.h"

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/Tooling.h>

using namespace tau::frontend;

namespace {
class AIRGenASTConsumer final : public clang::ASTConsumer {
public:
  AIRGenASTConsumer(clang::CompilerInstance &CI, OwningModuleRef &ToInit,
                    mlir::MLIRContext &Context)
      : ToInit(ToInit), Context(Context) {}

  void HandleTranslationUnit(clang::ASTContext &CContext) {
    ToInit = AIRGenerator::generate(Context, CContext);
  }

private:
  OwningModuleRef &ToInit;
  mlir::MLIRContext &Context;
};

class AIRGenActionImpl final : public clang::SyntaxOnlyAction {
public:
  AIRGenActionImpl(OwningModuleRef &ToInit, mlir::MLIRContext &Context)
      : ToInit(ToInit), Context(Context) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<AIRGenASTConsumer>(CI, ToInit, Context);
  }

private:
  OwningModuleRef &ToInit;
  mlir::MLIRContext &Context;
};
} // end anonymous namespace

std::unique_ptr<clang::FrontendAction> AIRGenAction::create() {
  return std::make_unique<AIRGenActionImpl>(Module, Context);
}
