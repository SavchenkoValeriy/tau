#include "tau/Frontend/Clang/AIRGenerator.h"

#include <mlir/IR/Builders.h>

using namespace tau::frontend;

namespace {

class GeneratorImpl {
public:
  GeneratorImpl(mlir::MLIRContext &MContext, clang::ASTContext &Context)
      : Builder(&MContext), Context(Context) {}
  mlir::ModuleOp generateModule();

private:
  mlir::ModuleOp Module;
  mlir::Builder Builder;
  clang::ASTContext &Context;
};

} // end anonymous namespace

mlir::ModuleOp GeneratorImpl::generateModule() {
  Module = mlir::ModuleOp::create(Builder.getUnknownLoc());
  return Module;
}

mlir::OwningModuleRef AIRGenerator::generate(mlir::MLIRContext &MContext,
                                             clang::ASTContext &Context) {
  GeneratorImpl ActualGenerator(MContext, Context);
  return ActualGenerator.generateModule();
}
