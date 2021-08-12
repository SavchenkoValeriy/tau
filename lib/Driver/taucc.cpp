#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <mlir/IR/MLIRContext.h>
#include <memory>

using namespace clang;
using namespace llvm;

cl::OptionCategory TauCategory("tau compiler options");

class AIRGenASTConsumer final : public clang::ASTConsumer {
public:
  AIRGenASTConsumer(clang::CompilerInstance &CI) {}

  void HandleTranslationUnit(clang::ASTContext &Context) {
    mlir::MLIRContext MContext;
  }
};

class AIRGenAction final : public clang::SyntaxOnlyAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<AIRGenASTConsumer>(CI);
  }
};

class AIRGenActionFactory final : public tooling::FrontendActionFactory {
public:
  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<AIRGenAction>();
  }
};


int main(int Argc, const char **Argv) {
  tooling::CommonOptionsParser OptionsParser(Argc, Argv, TauCategory);
  tooling::ClangTool Tool(OptionsParser.getCompilations(),
                          OptionsParser.getSourcePathList());
  AIRGenActionFactory F;
  return Tool.run(&F);
}
