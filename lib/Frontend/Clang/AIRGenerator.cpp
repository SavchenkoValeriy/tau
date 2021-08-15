#include "tau/Frontend/Clang/AIRGenerator.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/Basic/SourceManager.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>

using namespace clang;
using namespace mlir;
using namespace tau::frontend;

namespace {

using ShortString = SmallString<32>;

class TopLevelGenerator {
public:
  TopLevelGenerator(MLIRContext &MContext, ASTContext &Context)
      : Builder(&MContext), Context(Context) {}
  ModuleOp generateModule();

  void generateDecl(const Decl *D);
  void generateFunction(const FunctionDecl *F);
  void generateNamespace(const NamespaceDecl *NS);
  void generateRecord(const RecordDecl *RD);

  mlir::Type getType(clang::QualType);
  mlir::Type getBuiltinType(clang::QualType);

  ShortString getFullyQualifiedName(const NamedDecl *ND);

  mlir::Location loc(clang::SourceRange);
  mlir::Location loc(clang::SourceLocation);

  ModuleOp &getModule() { return Module; }
  OpBuilder &getBuilder() { return Builder; }
  ASTContext &getContext() { return Context; }

private:
  ModuleOp Module;
  OpBuilder Builder;
  ASTContext &Context;
};

class FunctionGenerator
    : public clang::ConstStmtVisitor<FunctionGenerator, mlir::Value> {
public:
  static void generate(FuncOp &ToGenerate, const FunctionDecl &Original,
                       TopLevelGenerator &Parent) {
    if (!Original.hasBody())
      return;

    FunctionGenerator G{ToGenerate, Original, Parent};
    G.generate();
  }

  mlir::Value VisitCompoundStmt(const CompoundStmt *Compound) {
    for (const auto *Child : Compound->body())
      Visit(Child);
    return nullptr;
  }

  mlir::Value VisitReturnStmt(const ReturnStmt *Return) {
    mlir::Value Operand = Visit(Return->getRetValue());
    ReturnOp Result = Builder.create<mlir::ReturnOp>(
        Parent.loc(Return->getSourceRange()),
        Operand ? llvm::makeArrayRef(Operand) : ArrayRef<mlir::Value>());
    return nullptr;
  }

  mlir::Value VisitIntegerLiteral(const IntegerLiteral *Literal) {
    return Builder.create<ConstantIntOp>(
        Parent.loc(Literal->getSourceRange()),
        *Literal->getValue().getRawData(),
        Context.getIntWidth(Literal->getType()));
  }

private:
  FunctionGenerator(FuncOp &ToGenerate, const FunctionDecl &Original,
                    TopLevelGenerator &Parent)
      : Target(ToGenerate), Original(Original), Parent(Parent),
        Context(Parent.getContext()), Builder(Parent.getBuilder()) {}

  void generate() {
    Block *Entry = Target.addEntryBlock();
    Builder.setInsertionPointToStart(Entry);
    Visit(Original.getBody());
  }

  FuncOp &Target;
  const FunctionDecl &Original;
  TopLevelGenerator &Parent;
  ASTContext &Context;
  OpBuilder &Builder;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                                  Utiliities
//===----------------------------------------------------------------------===//

ShortString TopLevelGenerator::getFullyQualifiedName(const NamedDecl *ND) {
  ShortString Result;
  llvm::raw_svector_ostream SS{Result};

  PrintingPolicy Policy = Context.getPrintingPolicy();
  Policy.TerseOutput = true;
  Policy.FullyQualifiedName = true;
  Policy.PrintCanonicalTypes = true;

  ND->print(SS, Policy);

  return Result;
}

mlir::Location TopLevelGenerator::loc(clang::SourceRange R) {
  return Builder.getFusedLoc({loc(R.getBegin()), loc(R.getEnd())});
}

mlir::Location TopLevelGenerator::loc(clang::SourceLocation L) {
  const SourceManager &SM = Context.getSourceManager();
  return Builder.getFileLineColLoc(Builder.getIdentifier(SM.getFilename(L)),
                                   SM.getSpellingLineNumber(L),
                                   SM.getSpellingColumnNumber(L));
}

//===----------------------------------------------------------------------===//
//                                    Types
//===----------------------------------------------------------------------===//

mlir::Type TopLevelGenerator::getType(clang::QualType T) {
  switch (T->getTypeClass()) {
  case clang::Type::Builtin:
    return getBuiltinType(T);
  default:
    return Builder.getNoneType();
  }
}

mlir::Type TopLevelGenerator::getBuiltinType(clang::QualType T) {
  if (T->isIntegralOrEnumerationType())
    return Builder.getIntegerType(Context.getIntWidth(T),
                                  T->isSignedIntegerOrEnumerationType());
  if (T->isFloat128Type())
    return Builder.getF128Type();

  if (T->isSpecificBuiltinType(BuiltinType::LongDouble))
    return Builder.getF128Type();

  if (T->isSpecificBuiltinType(BuiltinType::Double))
    return Builder.getF64Type();

  if (T->isFloat16Type())
    return Builder.getF16Type();

  return Builder.getNoneType();
}

//===----------------------------------------------------------------------===//
//                             Top-level generators
//===----------------------------------------------------------------------===//

ModuleOp TopLevelGenerator::generateModule() {
  Context.getTranslationUnitDecl()->dump();
  Module = ModuleOp::create(Builder.getUnknownLoc());

  const auto *TU = Context.getTranslationUnitDecl();
  for (const auto *TUDecl : TU->decls())
    generateDecl(TUDecl);

  return Module;
}

void TopLevelGenerator::generateDecl(const Decl *D) {
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

void TopLevelGenerator::generateNamespace(const NamespaceDecl *NS) {
  for (const auto *NSDecl : NS->decls())
    generateDecl(NSDecl);
}

void TopLevelGenerator::generateRecord(const RecordDecl *RD) {
  for (const auto *NestedDecl : RD->decls())
    generateDecl(NestedDecl);
}

void TopLevelGenerator::generateFunction(const FunctionDecl *F) {
  mlir::FunctionType T;
  ShortString Name = getFullyQualifiedName(F);

  llvm::SmallVector<mlir::Type, 4> ArgTypes;
  ArgTypes.reserve(F->getNumParams());
  for (const auto &Param : F->parameters()) {
    ArgTypes.push_back(getType(Param->getType()));
  }

  auto Result = FuncOp::create(
      loc(F->getSourceRange()), Name,
      Builder.getFunctionType(ArgTypes, getType(F->getReturnType())));
  FunctionGenerator::generate(Result, *F, *this);
  Module.push_back(Result);
}

OwningModuleRef AIRGenerator::generate(MLIRContext &MContext,
                                       ASTContext &Context) {
  MContext.loadDialect<mlir::StandardOpsDialect>();
  TopLevelGenerator ActualGenerator(MContext, Context);
  return ActualGenerator.generateModule();
}
