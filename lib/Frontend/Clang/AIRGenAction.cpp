#include "tau/Frontend/Clang/AIRGenAction.h"
#include "tau/Frontend/Clang/AIRGenerator.h"

#include <clang/AST/ASTContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/MultiplexConsumer.h>
#include <clang/Tooling/Tooling.h>
#include <memory>

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

class DumpASTConsumer final : public clang::ASTConsumer {
public:
  void HandleTranslationUnit(clang::ASTContext &Context) {
    Context.getTranslationUnitDecl()->dump();
  }
};

class AIRGenActionImpl final : public clang::SyntaxOnlyAction {
public:
  AIRGenActionImpl(OwningModuleRef &ToInit, mlir::MLIRContext &Context,
                   Options Opts)
      : ToInit(ToInit), Context(Context), Opts(Opts) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) override {
    std::vector<std::unique_ptr<clang::ASTConsumer>> Consumers;

    if (Opts.DumpAST)
      Consumers.emplace_back(std::make_unique<DumpASTConsumer>());

    Consumers.emplace_back(
        std::make_unique<AIRGenASTConsumer>(CI, ToInit, Context));

    return std::make_unique<clang::MultiplexConsumer>(std::move(Consumers));
  }

private:
  OwningModuleRef &ToInit;
  mlir::MLIRContext &Context;
  Options Opts;
};
} // end anonymous namespace

std::unique_ptr<clang::FrontendAction> AIRGenAction::create() {
  return std::make_unique<AIRGenActionImpl>(Module, Context, Opts);
}
