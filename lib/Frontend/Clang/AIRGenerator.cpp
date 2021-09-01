#include "tau/Frontend/Clang/AIRGenerator.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirTypes.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/None.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/Basic/SourceManager.h>
#include <immer/map.hpp>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>

using namespace clang;
using namespace mlir;
using namespace tau;
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
  mlir::Type getPointerType(clang::QualType);
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

  mlir::Value VisitCompoundStmt(const CompoundStmt *Compound);
  mlir::Value VisitReturnStmt(const ReturnStmt *Return);
  mlir::Value VisitDeclStmt(const DeclStmt *DS);

  mlir::Value VisitIntegerLiteral(const IntegerLiteral *Literal);
  mlir::Value VisitFloatingLiteral(const FloatingLiteral *Literal);
  mlir::Value VisitImplicitCastExpr(const ImplicitCastExpr *Cast);
  mlir::Value VisitDeclRefExpr(const DeclRefExpr *Ref);
  mlir::Value VisitBinaryOperator(const BinaryOperator *BinExpr);
  mlir::Value VisitUnaryOperator(const UnaryOperator *UnExpr);
  mlir::Value VisitParenExpr(const ParenExpr *Paren);

private:
  FunctionGenerator(FuncOp &ToGenerate, const FunctionDecl &Original,
                    TopLevelGenerator &Parent)
      : Target(ToGenerate), Original(Original), Parent(Parent),
        Context(Parent.getContext()), Builder(Parent.getBuilder()) {}

  void generate() {
    Block *Entry = Target.addEntryBlock();
    DeclScope ParamScope(Declarations);

    Builder.setInsertionPointToStart(Entry);
    for (const auto &[Param, BlockArg] :
         llvm::zip(Original.parameters(), Entry->getArguments())) {
      store(BlockArg, declare(Param), Parent.loc(Param->getSourceRange()));
    }

    Visit(Original.getBody());
    if (needToAddExtraReturn(Entry))
      Builder.create<mlir::ReturnOp>(Builder.getUnknownLoc());
  }

  bool needToAddExtraReturn(Block *BB) const {
    return Target.getNumResults() == 0 &&
           (BB->empty() || !BB->back().mightHaveTrait<OpTrait::IsTerminator>());
  }

  mlir::Value declare(const ValueDecl *D) {
    // TODO: support array types
    mlir::Value StackVar = Builder.create<air::AllocaOp>(
        Parent.loc(D->getSourceRange()),
        air::AirPointerType::get(Parent.getType(D->getType())), mlir::Value{});
    Declarations.insert(D, StackVar);
    return StackVar;
  }

  mlir::Value getPointer(const ValueDecl *D) const {
    return Declarations.lookup(D);
  }

  void store(mlir::Value V, const ValueDecl *D, Location Loc) {
    store(V, getPointer(D), Loc);
  }

  void store(mlir::Value What, mlir::Value Where, Location Loc) {
    Builder.create<air::StoreOp>(Loc, What, Where);
  }

  mlir::Value load(const ValueDecl *From, Location Loc) {
    return load(getPointer(From), Loc);
  }

  mlir::Value load(mlir::Value From, Location Loc) {
    return Builder.create<air::LoadOp>(Loc, From);
  }

  FuncOp &Target;
  const FunctionDecl &Original;
  TopLevelGenerator &Parent;
  ASTContext &Context;
  OpBuilder &Builder;
  using DeclMap = llvm::ScopedHashTable<const ValueDecl *, mlir::Value>;
  using DeclScope = DeclMap::ScopeTy;
  DeclMap Declarations;
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
  return mlir::FileLineColLoc::get(Builder.getIdentifier(SM.getFilename(L)),
                                   SM.getSpellingLineNumber(L),
                                   SM.getSpellingColumnNumber(L));
}

//===----------------------------------------------------------------------===//
//                                    Types
//===----------------------------------------------------------------------===//

mlir::Type TopLevelGenerator::getType(clang::QualType T) {
  switch (T->getTypeClass()) {
  case clang::Type::Pointer:
    return getPointerType(T);
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

  if (T->isSpecificBuiltinType(BuiltinType::Float))
    return Builder.getF32Type();

  if (T->isFloat16Type())
    return Builder.getF16Type();

  return Builder.getNoneType();
}

mlir::Type TopLevelGenerator::getPointerType(clang::QualType T) {
  mlir::Type NestedType = getType(T->getPointeeType());
  return air::AirPointerType::get(NestedType);
}

//===----------------------------------------------------------------------===//
//                             Top-level generators
//===----------------------------------------------------------------------===//

ModuleOp TopLevelGenerator::generateModule() {
  Module = ModuleOp::create(Builder.getUnknownLoc());

  const auto *TU = Context.getTranslationUnitDecl();
  for (const auto *TUDecl : TU->decls())
    generateDecl(TUDecl);

  if (failed(mlir::verify(Module))) {
    Module.emitError("Generated incorrect module");
    return nullptr;
  }
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
  case Decl::ClassTemplateSpecialization:
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

  mlir::TypeRange ReturnType = llvm::None;
  if (!F->getReturnType()->isVoidType())
    ReturnType = getType(F->getReturnType());

  auto Result = FuncOp::create(loc(F->getSourceRange()), Name,
                               Builder.getFunctionType(ArgTypes, ReturnType));
  FunctionGenerator::generate(Result, *F, *this);
  Module.push_back(Result);
}

OwningModuleRef AIRGenerator::generate(MLIRContext &MContext,
                                       ASTContext &Context) {
  MContext.loadDialect<mlir::StandardOpsDialect>();
  MContext.loadDialect<air::AirDialect>();
  TopLevelGenerator ActualGenerator(MContext, Context);
  return ActualGenerator.generateModule();
}

//===----------------------------------------------------------------------===//
//                                  Statements
//===----------------------------------------------------------------------===//

mlir::Value FunctionGenerator::VisitCompoundStmt(const CompoundStmt *Compound) {
  DeclScope Scope(Declarations);
  for (const auto *Child : Compound->body())
    Visit(Child);
  return nullptr;
}

mlir::Value FunctionGenerator::VisitReturnStmt(const ReturnStmt *Return) {
  mlir::Value Operand =
      Return->getRetValue() ? Visit(Return->getRetValue()) : mlir::Value{};
  ReturnOp Result = Builder.create<mlir::ReturnOp>(
      Parent.loc(Return->getSourceRange()),
      Operand ? llvm::makeArrayRef(Operand) : ArrayRef<mlir::Value>());
  return {};
}

mlir::Value FunctionGenerator::VisitDeclStmt(const DeclStmt *DS) {
  for (const auto *D : DS->decls())
    if (const auto *Var = dyn_cast<VarDecl>(D)) {
      // mlir::Value Init = Visit(Var->getInit());
      // TODO: handle situations with undefined initializations.
      declare(Var);
    }
  return {};
}

//===----------------------------------------------------------------------===//
//                                 Expressions
//===----------------------------------------------------------------------===//

mlir::Value
FunctionGenerator::VisitIntegerLiteral(const IntegerLiteral *Literal) {
  mlir::Type T = Parent.getBuiltinType(Literal->getType());
  assert(T.isa<IntegerType>());
  return Builder.create<air::ConstantIntOp>(
      Parent.loc(Literal->getSourceRange()), Literal->getValue(),
      T.cast<IntegerType>());
}

mlir::Value
FunctionGenerator::VisitFloatingLiteral(const FloatingLiteral *Literal) {
  mlir::Type T = Parent.getBuiltinType(Literal->getType());
  assert(T.isa<FloatType>());
  return Builder.create<ConstantFloatOp>(Parent.loc(Literal->getSourceRange()),
                                         Literal->getValue(),
                                         T.cast<FloatType>());
}

mlir::Value
FunctionGenerator::VisitImplicitCastExpr(const ImplicitCastExpr *Cast) {
  switch (Cast->getCastKind()) {
  case CastKind::CK_LValueToRValue:
    return load(Visit(Cast->getSubExpr()), Parent.loc(Cast->getSourceRange()));
  default:
    return {};
  }
}

mlir::Value FunctionGenerator::VisitDeclRefExpr(const DeclRefExpr *Ref) {
  return getPointer(Ref->getDecl());
}

mlir::Value FunctionGenerator::VisitUnaryOperator(const UnaryOperator *UnExpr) {
  mlir::Value Sub = Visit(UnExpr->getSubExpr());

  mlir::Location Loc = Parent.loc(UnExpr->getSourceRange());
  const bool IsInteger = UnExpr->getType()->isIntegralOrEnumerationType();

  switch (UnExpr->getOpcode()) {
  case UnaryOperatorKind::UO_AddrOf:
    // TODO: support taking address
    break;
  case UnaryOperatorKind::UO_Deref:
    // TODO: support dereference operation
    break;

  case UnaryOperatorKind::UO_PostInc:
  case UnaryOperatorKind::UO_PostDec:
  case UnaryOperatorKind::UO_PreInc:
  case UnaryOperatorKind::UO_PreDec:
    // TODO: support increment/decrement operations
    break;

  case UnaryOperatorKind::UO_Plus:
    // Unary plus is a no-op operation
    return Sub;
  case UnaryOperatorKind::UO_Minus:
    if (IsInteger)
      return Builder.create<air::NegIOp>(Loc, Sub);
    return Builder.create<mlir::NegFOp>(Loc, Sub);

  case UnaryOperatorKind::UO_Not:
    return Builder.create<air::NotOp>(Loc, Sub);

  case UnaryOperatorKind::UO_LNot:
    // TODO: support logical not operation
    break;

  case UnaryOperatorKind::UO_Real:
  case UnaryOperatorKind::UO_Imag:
    // TODO: support "__real expr"/"__imag expr" extension
    break;

  case UnaryOperatorKind::UO_Coawait:
    // TODO: support coroutine await
    break;
  case UnaryOperatorKind::UO_Extension:
    // TODO: support __extension__ marker
    break;
  }

  return {};
}

mlir::Value
FunctionGenerator::VisitBinaryOperator(const BinaryOperator *BinExpr) {
  mlir::Value LHS = Visit(BinExpr->getLHS());
  mlir::Value RHS = Visit(BinExpr->getRHS());

  mlir::Location Loc = Parent.loc(BinExpr->getSourceRange());

  bool IsInteger = BinExpr->getType()->isIntegralOrEnumerationType();

  mlir::Value Result;

  switch (BinExpr->getOpcode()) {
  case BinaryOperatorKind::BO_PtrMemD:
  case BinaryOperatorKind::BO_PtrMemI:
    // TODO: support pointer-to-member operators
    break;
  case BinaryOperatorKind::BO_MulAssign:
  case BinaryOperatorKind::BO_Mul:
    if (IsInteger)
      Result = Builder.create<mlir::MulIOp>(Loc, LHS, RHS);
    else
      Result = Builder.create<mlir::MulFOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_DivAssign:
  case BinaryOperatorKind::BO_Div:
  case BinaryOperatorKind::BO_RemAssign:
  case BinaryOperatorKind::BO_Rem:
    // TODO: support signed and unsigned division
    break;
  case BinaryOperatorKind::BO_AddAssign:
  case BinaryOperatorKind::BO_Add:
    if (IsInteger)
      Result = Builder.create<mlir::AddIOp>(Loc, LHS, RHS);
    else
      Result = Builder.create<mlir::AddFOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_SubAssign:
  case BinaryOperatorKind::BO_Sub:
    if (IsInteger)
      Result = Builder.create<mlir::SubIOp>(Loc, LHS, RHS);
    else
      Result = Builder.create<mlir::SubFOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_ShlAssign:
  case BinaryOperatorKind::BO_Shl:
    Result = Builder.create<mlir::ShiftLeftOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_ShrAssign:
  case BinaryOperatorKind::BO_Shr:
    // TODO: support unsigned shift right op
    Result = Builder.create<mlir::SignedShiftRightOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_Cmp:
    // TODO: support spaceship operator
    break;
  case BinaryOperatorKind::BO_LT:
  case BinaryOperatorKind::BO_GT:
  case BinaryOperatorKind::BO_LE:
  case BinaryOperatorKind::BO_GE:
  case BinaryOperatorKind::BO_EQ:
  case BinaryOperatorKind::BO_NE:
    // TODO: support comparison operators
    break;
  case BinaryOperatorKind::BO_AndAssign:
  case BinaryOperatorKind::BO_And:
    Result = Builder.create<mlir::AndOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_XorAssign:
  case BinaryOperatorKind::BO_Xor:
    Result = Builder.create<mlir::XOrOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_OrAssign:
  case BinaryOperatorKind::BO_Or:
    Result = Builder.create<mlir::OrOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_LAnd:
  case BinaryOperatorKind::BO_LOr:
    // TODO: support logical operators
    break;
  case BinaryOperatorKind::BO_Assign:
  case BinaryOperatorKind::BO_Comma:
    Result = RHS;
  }
  // TODO: support FP operations

  if (BinExpr->isAssignmentOp()) {
    // TODO: get L-value of LHS and store it there.
  }

  return Result;
}

mlir::Value FunctionGenerator::VisitParenExpr(const ParenExpr *Paren) {
  // We don't care abou parentheses at this point
  return Visit(Paren->getSubExpr());
}
