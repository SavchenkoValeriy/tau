#include "tau/Frontend/Clang/AIRGenerator.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/Basic/SourceManager.h>
#include <mlir/IR/Builders.h>

using namespace clang;
using namespace mlir;
using namespace tau::frontend;

namespace {

using ShortString = SmallString<32>;

class GeneratorImpl {
public:
  GeneratorImpl(MLIRContext &MContext, ASTContext &Context)
      : Builder(&MContext), Context(Context) {}
  ModuleOp generateModule();

  void generateDecl(const Decl *D);
  void generateFunction(const FunctionDecl *F);
  void generateNamespace(const NamespaceDecl *NS);
  void generateRecord(const RecordDecl *RD);

  ShortString getFullyQualifiedName(const NamedDecl *ND);

private:
  mlir::Location loc(clang::SourceRange);
  mlir::Location loc(clang::SourceLocation);

  ModuleOp Module;
  Builder Builder;
  ASTContext &Context;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                                  Utiliities
//===----------------------------------------------------------------------===//

ShortString GeneratorImpl::getFullyQualifiedName(const NamedDecl *ND) {
  ShortString Result;
  llvm::raw_svector_ostream SS{Result};

  PrintingPolicy Policy = Context.getPrintingPolicy();
  Policy.TerseOutput = true;
  Policy.FullyQualifiedName = true;
  Policy.PrintCanonicalTypes = true;

  ND->print(SS, Policy);

  return Result;
}

mlir::Location GeneratorImpl::loc(clang::SourceRange R) {
  return Builder.getFusedLoc({loc(R.getBegin()), loc(R.getEnd())});
}

mlir::Location GeneratorImpl::loc(clang::SourceLocation L) {
  const SourceManager &SM = Context.getSourceManager();
  return Builder.getFileLineColLoc(Builder.getIdentifier(SM.getFilename(L)),
                                   SM.getSpellingLineNumber(L),
                                   SM.getSpellingColumnNumber(L));
}

//===----------------------------------------------------------------------===//
//                             Top-level generators
//===----------------------------------------------------------------------===//

ModuleOp GeneratorImpl::generateModule() {
  Context.getTranslationUnitDecl()->dump();
  Module = ModuleOp::create(Builder.getUnknownLoc());

  const auto *TU = Context.getTranslationUnitDecl();
  for (const auto *TUDecl : TU->decls())
    generateDecl(TUDecl);

  return Module;
}

void GeneratorImpl::generateDecl(const Decl *D) {
  switch (D->getKind()) {
  case Decl::CXXMethod:
  case Decl::Function:
    generateFunction(cast<FunctionDecl>(D));
    break;
  case Decl::Namespace:
    generateNamespace(cast<NamespaceDecl>(D));
    break;
  case Decl::CXXRecord:
  case Decl::Record:
    generateRecord(cast<RecordDecl>(D));
    break;
  default:
    break;
  }
}

void GeneratorImpl::generateNamespace(const NamespaceDecl *NS) {
  for (const auto *NSDecl : NS->decls())
    generateDecl(NSDecl);
}

void GeneratorImpl::generateRecord(const RecordDecl *RD) {
  for (const auto *NestedDecl : RD->decls())
    generateDecl(NestedDecl);
}

void GeneratorImpl::generateFunction(const FunctionDecl *F) {
  mlir::FunctionType T;
  ShortString Name = getFullyQualifiedName(F);
  auto Result = FuncOp::create(loc(F->getSourceRange()), Name,
                               Builder.getFunctionType({}, {}));
  Module.push_back(Result);
}

OwningModuleRef AIRGenerator::generate(MLIRContext &MContext,
                                       ASTContext &Context) {
  GeneratorImpl ActualGenerator(MContext, Context);
  return ActualGenerator.generateModule();
}
