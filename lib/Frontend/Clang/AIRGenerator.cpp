#include "tau/Frontend/Clang/AIRGenerator.h"

#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirTypes.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/TemplateBase.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <iterator>
#include <utility>

using namespace clang;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::cf;
using namespace mlir::func;
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

  inline mlir::Type type(clang::QualType);
  template <class NodeTy> mlir::Type type(const NodeTy *Node) {
    return type(Node->getType());
  }

  mlir::Type getPointerType(clang::QualType);
  mlir::Type getBuiltinType(clang::QualType);
  mlir::Type getRecordType(clang::QualType);

  ShortString getFullyQualifiedName(const RecordDecl *RD) const;
  ShortString getFullyQualifiedName(const FunctionDecl *FD) const;
  ShortString getFullyQualifiedName(QualType T) const;
  void getFullyQualifiedName(QualType T, llvm::raw_svector_ostream &SS) const;

  inline mlir::Location loc(clang::SourceRange);
  inline mlir::Location loc(clang::SourceLocation);
  template <class NodeTy> mlir::Location loc(const NodeTy *Node) {
    return loc(Node->getSourceRange());
  }

  FuncOp getFunctionByDecl(const FunctionDecl *F) {
    return Functions.lookup(F);
  }

  ModuleOp &getModule() { return Module; }
  OpBuilder &getBuilder() { return Builder; }
  ASTContext &getContext() { return Context; }

private:
  DenseMap<const FunctionDecl *, FuncOp> Functions;
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

  mlir::Value VisitCallExpr(const CallExpr *Call);
  mlir::Value VisitIntegerLiteral(const IntegerLiteral *Literal);
  mlir::Value VisitFloatingLiteral(const FloatingLiteral *Literal);
  mlir::Value VisitDeclRefExpr(const DeclRefExpr *Ref);
  mlir::Value VisitBinaryOperator(const BinaryOperator *BinExpr);
  mlir::Value VisitUnaryOperator(const UnaryOperator *UnExpr);
  mlir::Value VisitParenExpr(const ParenExpr *Paren);

  mlir::Value VisitIfStmt(const IfStmt *If);
  mlir::Value VisitWhileStmt(const WhileStmt *While);

  mlir::Value VisitCXXStaticCastExpr(const CXXStaticCastExpr *Cast);
  mlir::Value VisitCStyleCastExpr(const CStyleCastExpr *Cast);
  mlir::Value VisitImplicitCastExpr(const ImplicitCastExpr *Cast);

  mlir::Value generateIncDec(mlir::Location Loc, mlir::Value Var, bool IsPre,
                             bool IsInc);

  template <class IntOp, class FloatOp, class... Args>
  mlir::Value builtinOp(mlir::Type OpType, Args &&...Rest);
  template <class SignedIntOp, class UnsignedIntOp, class FloatOp,
            class... Args>
  mlir::Value builtinOp(mlir::Type OpType, Args &&...Rest);
  template <class SignedIntOp, class UnsignedIntOp, class... Args>
  mlir::Value builtinIOp(mlir::Type OpType, Args &&...Rest);

private:
  FunctionGenerator(FuncOp &ToGenerate, const FunctionDecl &Original,
                    TopLevelGenerator &Parent)
      : Target(ToGenerate), Original(Original), Parent(Parent),
        Context(Parent.getContext()), Builder(Parent.getBuilder()) {}

  void generate() {
    Entry = Target.addEntryBlock();
    Exit = Target.addBlock();
    handleExit();

    DeclScope ParamScope(Declarations);

    Builder.setInsertionPointToStart(Entry);
    for (const auto &[Param, BlockArg] :
         llvm::zip(Original.parameters(), Entry->getArguments())) {
      declare(Param, BlockArg);
    }

    Visit(Original.getBody());
    Block *CurrentBlock = Builder.getBlock();

    if (needToAddExtraReturn(CurrentBlock))
      Builder.create<mlir::cf::BranchOp>(Builder.getUnknownLoc(), Exit);

    if (CurrentBlock->hasNoPredecessors() && CurrentBlock != Entry)
      CurrentBlock->erase();
  }

  void handleExit() {
    Builder.setInsertionPointToStart(Exit);
    mlir::Location Loc = Builder.getUnknownLoc();
    if (Target.getNumResults() == 0)
      Builder.create<ReturnOp>(Loc);
    else {
      BlockArgument ReturnValue =
          Exit->addArgument(Target.getResultTypes()[0], Loc);
      Builder.create<ReturnOp>(Loc, ReturnValue);
    }
  }

  bool needToAddExtraReturn(Block *BB) const {
    return Target.getNumResults() == 0 && hasNoTerminator(BB);
  }

  static bool hasNoTerminator(Block *BB) {
    return BB->empty() || !BB->back().mightHaveTrait<OpTrait::IsTerminator>();
  }

  mlir::Value declare(const ValueDecl *D, mlir::Value InitValue) {
    // Here we want to point to the value name, not its type.
    // For this reason, we use D->getLocation() as the start
    // location instead of D->getBeginLoc().
    mlir::Location Loc = loc(SourceRange{D->getLocation(), D->getEndLoc()});
    mlir::Type T = type(D);

    if (!InitValue) {
      InitValue = Builder.create<air::UndefOp>(Loc, T);
    }

    mlir::Value Result;
    if (D->getType()->isLValueReferenceType()) {
      Result = Builder.create<air::RefOp>(Loc, InitValue);
    } else {
      // TODO: support array types
      Result = Builder.create<air::AllocaOp>(Loc, air::PointerType::get(T),
                                             mlir::Value{});
      store(InitValue, Result, Loc);
    }
    Declarations.insert(D, Result);
    return Result;
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

  mlir::Value cast(mlir::Location Loc, mlir::Value Value, IntegerType To);

  using Values = SmallVector<mlir::Value, 8>;
  template <class RangeTy> Values visitRange(RangeTy &&Range) {
    Values Result;
    llvm::transform(Range, std::back_inserter(Result),
                    [this](const Stmt *S) { return Visit(S); });
    return Result;
  }

  template <class SmthWithLoc> mlir::Location loc(SmthWithLoc &&Object) {
    return Parent.loc(std::forward<SmthWithLoc>(Object));
  }
  template <class SmthWithType> mlir::Type type(SmthWithType &&Object) {
    return Parent.type(std::forward<SmthWithType>(Object));
  }

  FuncOp &Target;
  const FunctionDecl &Original;
  TopLevelGenerator &Parent;
  ASTContext &Context;
  OpBuilder &Builder;
  using DeclMap = llvm::ScopedHashTable<const ValueDecl *, mlir::Value>;
  using DeclScope = DeclMap::ScopeTy;
  DeclMap Declarations;
  Block *Entry, *Exit;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                                  Utiliities
//===----------------------------------------------------------------------===//

ShortString
TopLevelGenerator::getFullyQualifiedName(const RecordDecl *RD) const {
  ShortString Result;
  llvm::raw_svector_ostream SS{Result};

  SS << RD->getQualifiedNameAsString();
  if (const auto *AsTemplateSpecialization =
          dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
    PrintingPolicy Policy = Context.getPrintingPolicy();
    Policy.TerseOutput = true;
    Policy.FullyQualifiedName = true;
    Policy.PrintCanonicalTypes = true;

    SS << "<";
    llvm::interleaveComma(AsTemplateSpecialization->getTemplateArgs().asArray(),
                          SS,
                          [&SS, &Policy, this](const TemplateArgument &Arg) {
                            Arg.print(Policy, SS, true);
                          });
    SS << ">";
  }

  return Result;
}

ShortString
TopLevelGenerator::getFullyQualifiedName(const FunctionDecl *FD) const {
  ShortString Result;
  llvm::raw_svector_ostream SS{Result};

  getFullyQualifiedName(FD->getReturnType(), SS);
  SS << " ";
  SS << FD->getQualifiedNameAsString();
  SS << "(";
  llvm::interleaveComma(FD->parameters(), SS,
                        [&SS, this](const ParmVarDecl *PD) {
                          getFullyQualifiedName(PD->getType(), SS);
                        });
  SS << ")";

  return Result;
}

ShortString TopLevelGenerator::getFullyQualifiedName(QualType T) const {
  ShortString Result;
  llvm::raw_svector_ostream SS{Result};
  getFullyQualifiedName(T, SS);
  return Result;
}

void TopLevelGenerator::getFullyQualifiedName(
    QualType T, llvm::raw_svector_ostream &SS) const {
  PrintingPolicy Policy = Context.getPrintingPolicy();
  Policy.TerseOutput = true;
  Policy.FullyQualifiedName = true;
  Policy.PrintCanonicalTypes = true;

  T.print(SS, Policy);
}

mlir::Location TopLevelGenerator::loc(clang::SourceRange R) {
  return Builder.getFusedLoc({loc(R.getBegin()), loc(R.getEnd())});
}

mlir::Location TopLevelGenerator::loc(clang::SourceLocation L) {
  const SourceManager &SM = Context.getSourceManager();
  return mlir::FileLineColLoc::get(Builder.getStringAttr(SM.getFilename(L)),
                                   SM.getSpellingLineNumber(L),
                                   SM.getSpellingColumnNumber(L));
}

//===----------------------------------------------------------------------===//
//                                    Types
//===----------------------------------------------------------------------===//

mlir::Type TopLevelGenerator::type(clang::QualType T) {
  switch (T->getTypeClass()) {
  case clang::Type::LValueReference:
  case clang::Type::Pointer:
    return getPointerType(T);
  case clang::Type::Builtin:
    return getBuiltinType(T);
  case clang::Type::Record:
    return getRecordType(T);
  case clang::Type::TemplateSpecialization:
    // TODO: revise this in the future, we drop arguments out
    //       of the name.
    return getRecordType(
        T->castAs<clang::TemplateSpecializationType>()->desugar());
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
  const mlir::Type NestedType = type(T->getPointeeType());
  return air::PointerType::get(NestedType);
}

mlir::Type TopLevelGenerator::getRecordType(clang::QualType T) {
  return air::RecordRefType::get(Builder.getContext(),
                                 getFullyQualifiedName(T));
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
  const auto *Def = RD->getDefinition();
  if (Def == RD) {
    llvm::SmallVector<air::RecordField, 4> Fields;
    for (const auto *Field : RD->fields()) {
      Fields.push_back({Field->getName(), type(Field->getType())});
    }
    air::RecordType T = air::RecordType::get(Builder.getContext(), Fields);
    const auto D =
        air::RecordDefOp::create(loc(RD), getFullyQualifiedName(RD), T);
    Module.push_back(D);
  } else if (Def == nullptr) {
    // TODO: there might be multiple forward declarations
    //       of the same type, we should keep only one
    const auto D =
        air::RecordDeclOp::create(loc(RD), getFullyQualifiedName(RD));
    Module.push_back(D);
  }
  for (const auto *NestedDecl : RD->decls())
    generateDecl(NestedDecl);
}

void TopLevelGenerator::generateFunction(const FunctionDecl *F) {
  mlir::FunctionType T;
  ShortString Name = getFullyQualifiedName(F);

  llvm::SmallVector<mlir::Type, 4> ArgTypes;
  ArgTypes.reserve(F->getNumParams());
  for (const auto &Param : F->parameters()) {
    ArgTypes.push_back(type(Param));
  }

  mlir::TypeRange ReturnType = llvm::None;
  if (!F->getReturnType()->isVoidType())
    ReturnType = type(F->getReturnType());

  auto Result = FuncOp::create(loc(F), Name,
                               Builder.getFunctionType(ArgTypes, ReturnType));
  // It is important to do this BEFORE we generate function's body
  // because of possible recursions.
  Functions[F] = Result;
  FunctionGenerator::generate(Result, *F, *this);

  Module.push_back(Result);
}

OwningModuleRef AIRGenerator::generate(MLIRContext &MContext,
                                       ASTContext &Context) {
  MContext.loadDialect<mlir::func::FuncDialect>();
  MContext.loadDialect<mlir::cf::ControlFlowDialect>();
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
  Builder.create<BranchOp>(loc(Return), Exit,
                           Operand ? llvm::makeArrayRef(Operand)
                                   : ArrayRef<mlir::Value>());
  return {};
}

mlir::Value FunctionGenerator::VisitDeclStmt(const DeclStmt *DS) {
  for (const auto *D : DS->decls())
    if (const auto *Var = dyn_cast<VarDecl>(D)) {
      mlir::Location Loc = loc(Var);
      mlir::Value Init;
      if (Var->getInit()) {
        Init = Visit(Var->getInit());
      }
      mlir::Value Address = declare(Var, Init);
    }
  return {};
}

//===----------------------------------------------------------------------===//
//                                 Expressions
//===----------------------------------------------------------------------===//

mlir::Value FunctionGenerator::VisitCallExpr(const CallExpr *Call) {
  if (const FunctionDecl *DirectCallee = Call->getDirectCallee()) {

    const FuncOp Callee = Parent.getFunctionByDecl(DirectCallee);
    const Values Args = visitRange(Call->arguments());

    auto Result = Builder.create<CallOp>(loc(Call), Callee, Args);

    // Return result of the call if the call has any.
    // We couldn't have more than one result by construction.
    return Result.getNumResults() != 0 ? Result.getResult(0) : mlir::Value{};
  }
  // TODO: support indirect calls
  return {};
}

mlir::Value
FunctionGenerator::VisitIntegerLiteral(const IntegerLiteral *Literal) {
  mlir::Type T = type(Literal);
  assert(T.isa<IntegerType>());
  return Builder.create<air::ConstantIntOp>(loc(Literal), Literal->getValue(),
                                            T.cast<IntegerType>());
}

mlir::Value
FunctionGenerator::VisitFloatingLiteral(const FloatingLiteral *Literal) {
  mlir::Type T = type(Literal);
  assert(T.isa<FloatType>());
  return Builder.create<air::ConstantFloatOp>(loc(Literal), Literal->getValue(),
                                              T.cast<FloatType>());
}

mlir::Value FunctionGenerator::VisitDeclRefExpr(const DeclRefExpr *Ref) {
  return getPointer(Ref->getDecl());
}

mlir::Value FunctionGenerator::VisitUnaryOperator(const UnaryOperator *UnExpr) {
  mlir::Value Sub = Visit(UnExpr->getSubExpr());

  mlir::Location Loc = loc(UnExpr);
  mlir::Type ResultType = Sub.getType();

  switch (UnExpr->getOpcode()) {
  case UnaryOperatorKind::UO_AddrOf:
  case UnaryOperatorKind::UO_Deref:
    // Since for l-values, we actually use pointers and use l-value to r-value
    // casts for loads, we are going to be just fine using that here.
    return Sub;

  case UnaryOperatorKind::UO_PostInc:
  case UnaryOperatorKind::UO_PostDec:
  case UnaryOperatorKind::UO_PreInc:
  case UnaryOperatorKind::UO_PreDec:
    return generateIncDec(Loc, Sub, UnExpr->isPrefix(),
                          UnExpr->isIncrementOp());

  case UnaryOperatorKind::UO_Plus:
    // Unary plus is a no-op operation
    return Sub;
  case UnaryOperatorKind::UO_Minus:
    return builtinOp<air::NegIOp, NegFOp>(ResultType, Loc, Sub);

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

mlir::Value FunctionGenerator::generateIncDec(mlir::Location Loc,
                                              mlir::Value Var, bool IsPre,
                                              bool IsInc) {
  mlir::Value StoredValue = load(Var, Loc);
  mlir::Type ValueType = StoredValue.getType();
  const bool IsInteger = ValueType.isa<IntegerType>();

  mlir::Value Result;

  if (IsInteger) {
    mlir::Value One = Builder.create<air::ConstantIntOp>(
        Loc, 1, ValueType.cast<IntegerType>());

    if (IsInc)
      Result = Builder.create<air::AddIOp>(Loc, StoredValue, One);
    else
      Result = Builder.create<air::SubIOp>(Loc, StoredValue, One);
  } else {
    mlir::Value One = Builder.create<air::ConstantFloatOp>(
        Loc, 1.0, ValueType.cast<FloatType>());

    if (IsInc)
      Result = Builder.create<AddFOp>(Loc, StoredValue, One);
    else
      Result = Builder.create<SubFOp>(Loc, StoredValue, One);
  }

  // No matter whar - store the new value.
  store(Result, Var, Loc);

  // Prefix increment/decrement returns an l-value, and that is
  // what we are going to do here.
  // For postfix operations, we simply return the value that was
  // in the variable prior to the operation.
  return IsPre ? Var : StoredValue;
}

mlir::Value
FunctionGenerator::VisitBinaryOperator(const BinaryOperator *BinExpr) {
  mlir::Value LHS = Visit(BinExpr->getLHS());
  mlir::Value RHS = Visit(BinExpr->getRHS());

  mlir::Location Loc = loc(BinExpr);

  // Value representing the result of the operation
  mlir::Value Result;
  // Location where we should store the result if this is an
  // assignment operation.
  mlir::Value LocToStore;

  if (BinExpr->isCompoundAssignmentOp()) {
    // If this is a compound assignment operator (i.e. +=/-=/etc.)
    // we should first store the result into the LHS...
    LocToStore = LHS;
    // ...and also load the value from it to participate in the
    // operation.
    LHS = load(LHS, LHS.getLoc());
  }

  mlir::Type ResultType = LHS.getType();

  switch (BinExpr->getOpcode()) {
  case BinaryOperatorKind::BO_PtrMemD:
  case BinaryOperatorKind::BO_PtrMemI:
    // TODO: support pointer-to-member operators
    break;
  case BinaryOperatorKind::BO_MulAssign:
  case BinaryOperatorKind::BO_Mul:
    Result = builtinOp<air::MulIOp, MulFOp>(ResultType, Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_DivAssign:
  case BinaryOperatorKind::BO_Div:
    Result = builtinOp<air::SignedDivIOp, air::UnsignedDivIOp, DivFOp>(
        ResultType, Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_RemAssign:
  case BinaryOperatorKind::BO_Rem:
    Result = builtinIOp<air::SignedRemIOp, air::UnsignedRemIOp>(ResultType, Loc,
                                                                LHS, RHS);
    break;
  case BinaryOperatorKind::BO_AddAssign:
  case BinaryOperatorKind::BO_Add:
    Result = builtinOp<air::AddIOp, AddFOp>(ResultType, Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_SubAssign:
  case BinaryOperatorKind::BO_Sub:
    Result = builtinOp<air::SubIOp, SubFOp>(ResultType, Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_ShlAssign:
  case BinaryOperatorKind::BO_Shl:
    Result = Builder.create<air::ShiftLeftOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_ShrAssign:
  case BinaryOperatorKind::BO_Shr:
    Result = builtinIOp<air::ArithmeticShiftRightOp, air::LogicalShiftRightOp>(
        ResultType, Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_Cmp:
    // TODO: support spaceship operator
    break;
  case BinaryOperatorKind::BO_LT:
    Result = builtinOp<air::LessThanSIOp, air::LessThanUIOp, air::LessThanFOp>(
        ResultType, Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_GT:
    Result = builtinOp<air::GreaterThanSIOp, air::GreaterThanUIOp,
                       air::GreaterThanFOp>(ResultType, Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_LE:
    Result = builtinOp<air::LessThanOrEqualSIOp, air::LessThanOrEqualUIOp,
                       air::LessThanOrEqualFOp>(ResultType, Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_GE:
    Result = builtinOp<air::GreaterThanOrEqualSIOp, air::GreaterThanOrEqualUIOp,
                       air::GreaterThanOrEqualFOp>(ResultType, Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_EQ:
    Result = builtinOp<air::EqualIOp, air::EqualFOp>(ResultType, Loc, RHS, LHS);
    break;
  case BinaryOperatorKind::BO_NE:
    Result = builtinOp<air::NotEqualIOp, air::NotEqualFOp>(ResultType, Loc, RHS,
                                                           LHS);
    break;
  case BinaryOperatorKind::BO_AndAssign:
  case BinaryOperatorKind::BO_And:
    Result = Builder.create<air::AndOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_XorAssign:
  case BinaryOperatorKind::BO_Xor:
    Result = Builder.create<air::XOrOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_OrAssign:
  case BinaryOperatorKind::BO_Or:
    Result = Builder.create<air::OrOp>(Loc, LHS, RHS);
    break;
  case BinaryOperatorKind::BO_LAnd:
  case BinaryOperatorKind::BO_LOr:
    // TODO: support logical operators
    break;
  case BinaryOperatorKind::BO_Assign:
    LocToStore = LHS;
    [[fallthrough]];
  case BinaryOperatorKind::BO_Comma:
    Result = RHS;
  }

  if (BinExpr->isAssignmentOp()) {
    store(Result, LocToStore, Loc);
  }

  return Result;
}

template <class IntOp, class FloatOp, class... Args>
mlir::Value FunctionGenerator::builtinOp(mlir::Type OpType, Args &&...Rest) {
  if (OpType.isa<IntegerType>())
    return Builder.create<IntOp>(std::forward<Args>(Rest)...);
  return Builder.create<FloatOp>(std::forward<Args>(Rest)...);
}

template <class SignedIntOp, class UnsignedIntOp, class FloatOp, class... Args>
mlir::Value FunctionGenerator::builtinOp(mlir::Type OpType, Args &&...Rest) {
  if (OpType.isa<IntegerType>())
    return builtinIOp<SignedIntOp, UnsignedIntOp>(OpType,
                                                  std::forward<Args>(Rest)...);
  return Builder.create<FloatOp>(std::forward<Args>(Rest)...);
}

template <class SignedIntOp, class UnsignedIntOp, class... Args>
mlir::Value FunctionGenerator::builtinIOp(mlir::Type OpType, Args &&...Rest) {
  if (OpType.isSignedInteger())
    return Builder.create<SignedIntOp>(std::forward<Args>(Rest)...);
  return Builder.create<UnsignedIntOp>(std::forward<Args>(Rest)...);
}

mlir::Value FunctionGenerator::VisitParenExpr(const ParenExpr *Paren) {
  // We don't care abou parentheses at this point
  return Visit(Paren->getSubExpr());
}

//===----------------------------------------------------------------------===//
//                               Cast expressions
//===----------------------------------------------------------------------===//

mlir::Value
FunctionGenerator::VisitCXXStaticCastExpr(const CXXStaticCastExpr *Cast) {
  mlir::Value Sub = Visit(Cast->getSubExpr());
  mlir::Location Loc = loc(Cast);
  mlir::Type To = type(Cast);

  switch (Cast->getCastKind()) {
  case CastKind::CK_NoOp:
    return Sub;
  default:
    return {};
  }
}

mlir::Value FunctionGenerator::VisitCStyleCastExpr(const CStyleCastExpr *Cast) {
  mlir::Value Sub = Visit(Cast->getSubExpr());
  mlir::Location Loc = loc(Cast);
  mlir::Type To = type(Cast);

  switch (Cast->getCastKind()) {
  case CastKind::CK_NoOp:
    return Sub;
  default:
    return {};
  }
}

mlir::Value
FunctionGenerator::VisitImplicitCastExpr(const ImplicitCastExpr *Cast) {
  mlir::Value Sub = Visit(Cast->getSubExpr());
  mlir::Location Loc = loc(Cast);
  mlir::Type To = type(Cast);

  switch (Cast->getCastKind()) {
  case CastKind::CK_LValueToRValue:
    return load(Sub, Loc);
  case CastKind::CK_IntegralCast:
    return cast(Loc, Sub, To.cast<IntegerType>());
  case CastKind::CK_NullToPointer:
    // We don't actually care what kind of null is actually casted to pointer.
    // What we care is that it's a null.
    return Builder.create<air::NullOp>(Loc, To);
  default:
    return {};
  }
}

mlir::Value FunctionGenerator::cast(mlir::Location Loc, mlir::Value Value,
                                    IntegerType To) {
  IntegerType From = Value.getType().cast<IntegerType>();

  if (From.getWidth() < To.getWidth())
    return builtinIOp<air::SExtOp, air::ZExtOp>(From, Loc, To, Value);

  if (From.getWidth() > To.getWidth())
    return Builder.create<air::TruncateOp>(Loc, To, Value);

  return Builder.create<air::BitcastOp>(Loc, To, Value);
}

//===----------------------------------------------------------------------===//
//                           Control flow statements
//===----------------------------------------------------------------------===//

mlir::Value FunctionGenerator::VisitIfStmt(const IfStmt *If) {
  DeclScope IfVariableScope(Declarations);

  if (If->getConditionVariableDeclStmt())
    Visit(If->getConditionVariableDeclStmt());

  if (If->getInit())
    Visit(If->getInit());

  mlir::Value Cond = Visit(If->getCond());
  mlir::Location Loc = loc(If);

  // We need to remember this block to put a conditional branch
  // when all other blocks are ready.
  Block *IfBlock = Builder.getBlock();

  auto VisitBranch = [this](const Stmt *Branch) {
    // First, we create the block
    Block *BranchStart = Target.addBlock();
    // We generate all the code into that block
    Builder.setInsertionPointToStart(BranchStart);
    Visit(Branch);
    // And then we check what is the block after the code generation,
    // since the branch itself could've had a non-trivial CFG structure.
    Block *BranchEnd = Builder.getBlock();
    // Both "entry" and "exit" (so-to-speak) blocks of this branch
    // should be used to fully integrate new code into the function.
    return std::make_pair(BranchStart, BranchEnd);
  };

  // Handle true branch.
  auto [ThenStart, ThenEnd] = VisitBranch(If->getThen());

  // Handle false branch.
  Block *ElseStart = nullptr, *ElseEnd = nullptr;
  if (If->getElse())
    std::tie(ElseStart, ElseEnd) = VisitBranch(If->getElse());

  // We could've created this block a bit earlier, but we prefer
  // to keep topologocial ordering of blocks in a function whenever possible.
  Block *Next = Target.addBlock();
  auto AddBranchToNextIfNeeded = [&Next, this](Block *BB) {
    // Block can have early exits in them and might not need a branch to Next
    if (BB->hasNoSuccessors()) {
      Builder.setInsertionPointToEnd(BB);
      Builder.create<BranchOp>(Builder.getUnknownLoc(), Next);
    }
  };

  AddBranchToNextIfNeeded(ThenEnd);
  if (ElseEnd)
    AddBranchToNextIfNeeded(ElseEnd);
  else
    // No else means that we should head straight to Next.
    ElseStart = Next;

  // Now, when we have all the pieces in place, we can come back to the
  // original basic block...
  Builder.setInsertionPointToEnd(IfBlock);
  // ...create conditional branch...
  Builder.create<air::CondBranchOp>(Loc, Cond, ThenStart, ElseStart);

  // ...and continue with the rest of the function.
  Builder.setInsertionPointToStart(Next);

  return {};
}

mlir::Value FunctionGenerator::VisitWhileStmt(const WhileStmt *While) {
  DeclScope WhileVariableScope(Declarations);

  // Here we deal with three basic blocks:
  //   * basic block for the header of the loop
  Block *Header = Target.addBlock();
  //   * basic block for the loop's body
  Block *Body = Target.addBlock();
  //   * basic block for the code after the loop
  Block *Next = Target.addBlock();

  // First, let's generate the loop's header.
  Builder.create<BranchOp>(Builder.getUnknownLoc(), Header);
  Builder.setInsertionPointToStart(Header);

  // Loop variables...
  if (While->getConditionVariableDeclStmt())
    Visit(While->getConditionVariableDeclStmt());

  // ...and conditions belong to the header.
  mlir::Value Cond = Visit(While->getCond());
  mlir::Location Loc = loc(While);

  // If condition is true, we should proceed with the loop's
  // body, and skip it otherwise.
  Builder.create<air::CondBranchOp>(Loc, Cond, Body, Next);

  // Generate the body.
  Builder.setInsertionPointToStart(Body);
  Visit(While->getBody());

  // Let's check if the current block even needed.
  Block *Current = Builder.getBlock();
  if (Current->hasNoPredecessors())
    Current->erase();

  // And if yes, we might need to put a jump to the header block.
  else if (hasNoTerminator(Current)) {
    Builder.setInsertionPointToEnd(Current);
    Builder.create<BranchOp>(Builder.getUnknownLoc(), Header);
  }

  // All further generation should happen here.
  Builder.setInsertionPointToStart(Next);

  return {};
}
