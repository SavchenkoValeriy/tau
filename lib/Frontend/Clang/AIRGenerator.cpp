#include "tau/Frontend/Clang/AIRGenerator.h"

#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirTypes.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/TemplateBase.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TypeTraits.h>
#include <functional>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <iterator>
#include <stack>
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

class DeclarationsMap {
public:
  using KeyTy = const ValueDecl *;
  using ValueTy = mlir::Value;

  void insert(KeyTy Key, ValueTy Value) {
    const auto Pair = std::make_pair(Key, Value);
    Map.insert(Pair);
    currentScope().get().insert(Pair);
  }
  ValueTy lookup(KeyTy Key) const {
    if (const auto It = Map.find(Key); It != Map.end())
      return It->getSecond();

    return ValueTy();
  }

  class Scope {
  public:
    using ScopeHandler = llvm::function_ref<void(KeyTy, ValueTy)>;
    Scope(DeclarationsMap &Parent, ScopeHandler AtExit)
        : Parent(Parent), AtExit(AtExit) {
      Parent.ScopeStack.push(*this);
    }

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    Scope(Scope &&) = delete;
    Scope &operator=(Scope &&) = delete;

    ~Scope() {
      llvm::for_each(Values, [AtExit = AtExit](std::pair<KeyTy, ValueTy> Pair) {
        AtExit(Pair.first, Pair.second);
      });
      assert(&Parent.ScopeStack.top().get() == this &&
             "Inconsistent scope stack!");
      Parent.ScopeStack.pop();
    }

  private:
    void insert(std::pair<KeyTy, ValueTy> Value) { Values.push_back(Value); }

    DeclarationsMap &Parent;
    std::vector<std::pair<KeyTy, ValueTy>> Values;
    ScopeHandler AtExit;

    friend class DeclarationsMap;
  };

private:
  using ScopeRef = std::reference_wrapper<Scope>;

  ScopeRef currentScope() const {
    assert(!ScopeStack.empty() && "No scopes left!");
    return ScopeStack.top();
  }

  using MapTy = llvm::DenseMap<KeyTy, ValueTy>;
  MapTy Map;
  std::stack<ScopeRef> ScopeStack;
};

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
  void getFullyQualifiedName(QualType T, llvm::raw_svector_ostream &SS,
                             bool IncludeCV = false) const;

  inline mlir::Location loc(clang::SourceRange);
  inline mlir::Location loc(clang::SourceLocation);
  template <class NodeTy> mlir::Location loc(const NodeTy *Node) {
    return loc(Node->getSourceRange());
  }

  FuncOp getFunctionByDecl(const FunctionDecl *F) {
    return Functions.lookup(F);
  }

  air::RecordDefOp getRecordByType(clang::QualType T) {
    if (T->isRecordType())
      return getRecordByDecl(T->getAsRecordDecl());
    return {};
  }
  air::RecordDefOp getRecordByType(const RecordType *RT) {
    return getRecordByDecl(RT->getDecl());
  }
  air::RecordDefOp getRecordByDecl(const RecordDecl *RD) {
    if (const auto *Def = RD->getDefinition()) {
      return Records.lookup(Def);
    }
    return {};
  }

  ModuleOp &getModule() { return Module; }
  OpBuilder &getBuilder() { return Builder; }
  ASTContext &getContext() { return Context; }

private:
  DenseMap<const FunctionDecl *, FuncOp> Functions;
  DenseMap<const RecordDecl *, air::RecordDefOp> Records;
  ModuleOp Module;
  OpBuilder Builder;
  ASTContext &Context;
};

class FunctionGenerator
    : public clang::ConstStmtVisitor<FunctionGenerator, mlir::Value> {
public:
  static void generate(FuncOp &ToGenerate, const FunctionDecl &Original,
                       TopLevelGenerator &Parent) {
    if (!Original.hasBody()) {
      ToGenerate.setVisibility(mlir::SymbolTable::Visibility::Private);
      return;
    }

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
  mlir::Value VisitMemberExpr(const MemberExpr *Member);
  mlir::Value
  VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *Unary);
  mlir::Value
  VisitImplicitValueInitExpr(const ImplicitValueInitExpr *ImplicitValueInit);
  mlir::Value VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *CXXDefaultInit);

  mlir::Value VisitIfStmt(const IfStmt *If);
  mlir::Value VisitWhileStmt(const WhileStmt *While);
  mlir::Value VisitForStmt(const ForStmt *ForLoop);

  mlir::Value VisitCXXStaticCastExpr(const CXXStaticCastExpr *Cast);
  mlir::Value VisitCStyleCastExpr(const CStyleCastExpr *Cast);
  mlir::Value VisitImplicitCastExpr(const ImplicitCastExpr *Cast);

  mlir::Value VisitCXXThisExpr(const CXXThisExpr *ThisExpr);
  mlir::Value VisitCXXConstructExpr(const CXXConstructExpr *CtorCall);
  mlir::Value VisitCXXNewExpr(const CXXNewExpr *NewExpr);
  mlir::Value VisitInitListExpr(const InitListExpr *InitList);

  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *Literal);

  mlir::Value VisitExpr(const Expr *E);

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

    DeclScope ParamScope(Declarations, [](const ValueDecl *, mlir::Value) {});

    Builder.setInsertionPointToStart(Entry);
    auto Arguments = llvm::make_range(Entry->getArguments().begin(),
                                      Entry->getArguments().end());

    if (const auto *MD = dyn_cast<CXXMethodDecl>(&Original);
        MD && MD->isInstance()) {
      declareThis(*Arguments.begin());
      Arguments = llvm::drop_begin(Arguments);
    }
    for (const auto &[Param, BlockArg] :
         llvm::zip(Original.parameters(), Arguments)) {
      declare(Param, BlockArg);
    }

    if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(&Original)) {
      for (const auto *Init : Ctor->inits()) {
        if (!Init->isMemberInitializer())
          continue;

        const mlir::Location Loc = loc(Init);
        const auto *Field = Init->getMember();

        const mlir::Value FieldPtr = getMemberPointer(
            Loc, ThisParam, Field->getName(), type(Init->getInit()->getType()));
        init(FieldPtr, Init->getInit());
      }
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

    mlir::Type T = type(D);

    if (D->getType()->isLValueReferenceType()) {
      assert(InitValue);
      mlir::Value Result = Builder.create<air::RefOp>(loc(D), InitValue);
      Declarations.insert(D, Result);
      return Result;
    }

    mlir::Value Result = declare(D);
    store(InitValue, Result, Result.getLoc());
    return Result;
  }

  mlir::Value declare(const ValueDecl *D, const Expr *Init) {

    mlir::Type T = type(D);

    if (D->getType()->isLValueReferenceType()) {
      assert(Init);
      return declare(D, Visit(Init));
    }

    mlir::Value Result = declare(D);
    if (Init) {
      init(Result, Init);
    } else {
      mlir::Value InitValue = Builder.create<air::UndefOp>(Result.getLoc(), T);
      store(InitValue, Result, Result.getLoc());
    }
    return Result;
  }

  mlir::Value declare(const ValueDecl *D) {
    mlir::Value Result = alloca(D);
    Declarations.insert(D, Result);
    return Result;
  }

  mlir::Value alloca(const ValueDecl *D) {
    // Here we want to point to the value name, not its type.
    // For this reason, we use D->getLocation() as the start
    // location instead of D->getBeginLoc().
    mlir::Location Loc = loc(SourceRange{D->getLocation(), D->getEndLoc()});

    // TODO: support array types
    return Builder.create<air::AllocaOp>(Loc, air::PointerType::get(type(D)),
                                         mlir::Value{});
  }

  mlir::Value declareThis(mlir::Value InitValue) {
    // this pointer is not assignable, so we can express this as air.ref
    ThisParam = Builder.create<air::RefOp>(loc(&Original), InitValue);
    return ThisParam;
  }

  void init(mlir::Value Memory, const Expr *InitExpr);

  mlir::Value getPointer(const ValueDecl *D) const {
    return Declarations.lookup(D);
  }

  mlir::Value getMemberPointer(mlir::Location Loc, mlir::Value Base,
                               air::RecordField Member) {
    return getMemberPointer(Loc, Base, Member.Name, Member.Type);
  }

  mlir::Value getMemberPointer(mlir::Location Loc, mlir::Value Base,
                               llvm::StringRef Name, mlir::Type MemberTy) {
    mlir::Type PtrTy = air::PointerType::get(MemberTy);
    return Builder.create<air::GetFieldPtr>(Loc, PtrTy, Base, Name);
  }

  void store(mlir::Value V, const ValueDecl *D, Location Loc) {
    store(V, getPointer(D), Loc);
  }

  void store(mlir::Value What, mlir::Value Where, Location Loc) {
    assert(What && Where && "Don't store null values");
    Builder.create<air::StoreOp>(Loc, What, Where);
  }

  mlir::Value load(const ValueDecl *From, Location Loc) {
    return load(getPointer(From), Loc);
  }

  mlir::Value load(mlir::Value From, Location Loc) {
    return Builder.create<air::LoadOp>(Loc, From);
  }

  mlir::Value sizeOf(Location Loc, mlir::Type SizeType, mlir::Type Of) {
    return Builder.create<air::SizeOfOp>(Loc, SizeType, Of);
  }

  mlir::Value cast(mlir::Location Loc, mlir::Value Value, IntegerType To);
  mlir::Value bitcast(mlir::Location Loc, mlir::Value Value, mlir::Type To);

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
  using DeclScope = DeclarationsMap::Scope;
  DeclarationsMap Declarations;
  Block *Entry, *Exit;
  mlir::Value ThisParam;
  std::stack<mlir::Value> InitializedValues;
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

  llvm::SmallVector<clang::QualType> ParameterTypes;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (!MD->isStatic()) {
      ParameterTypes.push_back(MD->getThisType());
    }
  }
  ParameterTypes.reserve(FD->getNumParams());
  llvm::transform(FD->parameters(), std::back_inserter(ParameterTypes),
                  [](const ParmVarDecl *PD) { return PD->getType(); });

  getFullyQualifiedName(FD->getReturnType(), SS);
  SS << " ";
  SS << FD->getQualifiedNameAsString();
  SS << "(";
  llvm::interleaveComma(ParameterTypes, SS, [&SS, this](clang::QualType T) {
    getFullyQualifiedName(T, SS, true);
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

void TopLevelGenerator::getFullyQualifiedName(QualType T,
                                              llvm::raw_svector_ostream &SS,
                                              bool IncludeCV) const {
  if (!IncludeCV)
    T = T.getUnqualifiedType();

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
    return getRecordType(
        T->castAs<clang::TemplateSpecializationType>()->desugar());
  case clang::Type::Elaborated:
    return type(T->castAs<clang::ElaboratedType>()->desugar());
  case clang::Type::Typedef:
    return type(T->castAs<clang::TypedefType>()->desugar());
  default:
    // TODO: make unsupported type similarly to unsupported op
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

  if (T->isVoidType())
    return air::VoidType::get(Builder.getContext());

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
  }
  return Module;
}

void TopLevelGenerator::generateDecl(const Decl *D) {
  switch (D->getKind()) {
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
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
  case Decl::LinkageSpec:
    for (const auto *LSDecl : cast<LinkageSpecDecl>(D)->decls())
      generateDecl(LSDecl);
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
    llvm::SmallVector<air::RecordField, 8> Fields;
    llvm::SmallVector<mlir::Type, 4> Bases;

    if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (const auto &Base : CXXRD->bases()) {
        // TODO: handle virtual bases differently
        Bases.push_back(type(Base.getType()));
      }
    }

    for (const auto *Field : RD->fields()) {
      Fields.push_back({Field->getName(), type(Field->getType())});
    }
    air::RecordType T =
        air::RecordType::get(Builder.getContext(), Bases, Fields);
    const auto D =
        air::RecordDefOp::create(loc(RD), getFullyQualifiedName(RD), T);

    Records[RD] = D;
    Module.push_back(D);
  } else if (Def == nullptr && !RD->isImplicit()) {
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
  if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(F)) {
    // TODO: support r-value references and move constructors
    if (Ctor->isMoveConstructor())
      return;
  }
  ShortString Name = getFullyQualifiedName(F);

  llvm::SmallVector<mlir::Type, 4> ParamTypes;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(F); MD && MD->isInstance()) {
    ParamTypes.push_back(type(MD->getThisType()));
  }
  ParamTypes.reserve(F->getNumParams());
  for (const auto &Param : F->parameters()) {
    ParamTypes.push_back(type(Param));
  }

  mlir::TypeRange ReturnType = std::nullopt;
  if (!F->getReturnType()->isVoidType())
    ReturnType = type(F->getReturnType());

  auto FunctionType = Builder.getFunctionType(ParamTypes, ReturnType);
  auto Result = FuncOp::create(loc(F), Name, FunctionType);
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
  DeclScope Scope(Declarations, [](const ValueDecl *, mlir::Value) {});
  for (const auto *Child : Compound->body())
    Visit(Child);
  return nullptr;
}

mlir::Value FunctionGenerator::VisitReturnStmt(const ReturnStmt *Return) {
  mlir::Value Operand =
      Return->getRetValue() ? Visit(Return->getRetValue()) : mlir::Value{};
  Builder.create<BranchOp>(loc(Return), Exit,
                           Operand ? llvm::ArrayRef(Operand)
                                   : ArrayRef<mlir::Value>());
  return {};
}

mlir::Value FunctionGenerator::VisitDeclStmt(const DeclStmt *DS) {
  for (const auto *D : DS->decls())
    if (const auto *Var = dyn_cast<VarDecl>(D)) {
      mlir::Location Loc = loc(Var);
      declare(Var, Var->getInit());
    }
  return {};
}

//===----------------------------------------------------------------------===//
//                                 Expressions
//===----------------------------------------------------------------------===//

mlir::Value FunctionGenerator::VisitCallExpr(const CallExpr *Call) {
  if (const FunctionDecl *DirectCallee = Call->getDirectCallee()) {

    const FuncOp Callee = Parent.getFunctionByDecl(DirectCallee);
    Values Args;
    if (const auto *MethodCall = dyn_cast<CXXMemberCallExpr>(Call)) {
      if (const auto *ME = dyn_cast<MemberExpr>(MethodCall->getCallee())) {
        Args.push_back(Visit(ME->getBase()));
      }
    }
    Args.append(visitRange(Call->arguments()));

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
  // We don't care about parentheses at this point
  return Visit(Paren->getSubExpr());
}

mlir::Value FunctionGenerator::VisitMemberExpr(const MemberExpr *Member) {
  mlir::Value Result;

  if (const auto *MemberDecl = Member->getMemberDecl()) {
    Result = getMemberPointer(loc(Member), Visit(Member->getBase()),
                              MemberDecl->getName(), type(Member));
  }
  // TODO: support member access of static data and via pointers.
  return Result;
}

mlir::Value FunctionGenerator::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *Unary) {
  switch (Unary->getKind()) {
  case UnaryExprOrTypeTrait::UETT_SizeOf:
    return sizeOf(loc(Unary), type(Unary), type(Unary->getArgumentType()));
  default:
    return VisitExpr(Unary);
  }
}

mlir::Value FunctionGenerator::VisitImplicitValueInitExpr(
    const ImplicitValueInitExpr *ImplicitValueInit) {
  // TODO: check it for non-trivial types
  return Builder.create<air::UndefOp>(loc(ImplicitValueInit),
                                      type(ImplicitValueInit));
}

mlir::Value FunctionGenerator::VisitCXXDefaultInitExpr(
    const CXXDefaultInitExpr *CXXDefaultInit) {
  return Visit(CXXDefaultInit->getExpr());
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
  case CastKind::CK_DerivedToBase:
  case CastKind::CK_UncheckedDerivedToBase:
    return Builder.create<air::CastToBaseOp>(Loc, air::PointerType::get(To),
                                             Sub);
  case CastKind::CK_NoOp:
    return Sub;
  case CastKind::CK_BitCast:
    return bitcast(Loc, Sub, To);
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

  return bitcast(Loc, Value, To);
}

mlir::Value FunctionGenerator::bitcast(mlir::Location Loc, mlir::Value Value,
                                       mlir::Type To) {
  return Builder.create<air::BitcastOp>(Loc, To, Value);
}
//===----------------------------------------------------------------------===//
//                           C++-specific expressions
//===----------------------------------------------------------------------===//

mlir::Value FunctionGenerator::VisitCXXThisExpr(const CXXThisExpr *ThisExpr) {
  assert(ThisParam);
  return ThisParam;
}

mlir::Value
FunctionGenerator::VisitCXXConstructExpr(const CXXConstructExpr *CtorCall) {
  assert(!InitializedValues.empty());

  const FuncOp Callee = Parent.getFunctionByDecl(CtorCall->getConstructor());
  Values Args;
  Args.push_back(InitializedValues.top());
  Args.append(visitRange(CtorCall->arguments()));

  Builder.create<CallOp>(loc(CtorCall), Callee, Args);
  return {};
}

mlir::Value FunctionGenerator::VisitCXXNewExpr(const CXXNewExpr *NewExpr) {
  const auto Loc = loc(NewExpr);
  const auto Type = type(NewExpr);
  assert(Type.isa<air::PointerType>() &&
         "new expression should return a pointer");
  const auto Pointee = Type.cast<air::PointerType>().getElementType();

  // TODO: support array types
  if (NewExpr->isArray())
    return VisitExpr(NewExpr);

  auto PlacementArgs = visitRange(NewExpr->placement_arguments());
  mlir::Value Memory;
  if (!PlacementArgs.empty()) {
    if (auto MemType =
            PlacementArgs.front().getType().dyn_cast<air::PointerType>();
        MemType && PlacementArgs.size() == 1) {
      Memory = PlacementArgs.front();
      if (MemType != Type)
        Memory = bitcast(Loc, Memory, Type);

    } else if (const auto *New = NewExpr->getOperatorNew()) {
      const FuncOp Callee = Parent.getFunctionByDecl(New);
      Values Args;
      assert(New->getNumParams() != 0 &&
             "new operator should have at least 1 parameter");
      Args.push_back(
          sizeOf(Loc, type(New->getParamDecl(0)->getType()), Pointee));
      Args.append(PlacementArgs);

      auto NewOperatorCall = Builder.create<CallOp>(Loc, Callee, Args);
      assert(NewOperatorCall->getNumResults() != 0 &&
             "new operator should return a pointer");
      Memory = bitcast(Loc, NewOperatorCall->getResult(0), Type);
    }
  }

  if (!Memory)
    Memory =
        Builder.create<air::HeapAllocaOp>(Loc, type(NewExpr), mlir::Value{});

  InitializedValues.push(Memory);

  const auto RemoveInitializedValue = llvm::make_scope_exit([this, Memory]() {
    assert(!InitializedValues.empty() && InitializedValues.top() == Memory);
    InitializedValues.pop();
  });

  const auto *ConstructExpr = NewExpr->getConstructExpr();
  const auto *Initializer = NewExpr->getInitializer();

  if (ConstructExpr != nullptr)
    Visit(ConstructExpr);

  else if (Initializer != nullptr)
    store(Visit(Initializer), Memory, loc(Initializer));

  return Memory;
}

mlir::Value FunctionGenerator::VisitInitListExpr(const InitListExpr *InitList) {
  assert(!InitializedValues.empty());
  // TODO: Support array initialization
  const auto DesugaredType = InitList->getType().getDesugaredType(Context);
  switch (DesugaredType->getTypeClass()) {
  case clang::Type::Record: {
    air::RecordDefOp Def =
        Parent.getRecordByType(DesugaredType->castAs<RecordType>());
    if (!Def) {
      // This shouldn't really happen, we should've seen record
      // definition at this point.
      return {};
    }

    assert(Def.getRecordType().getFields().size() == InitList->inits().size() &&
           "Initializer list should cover all field, even the ones not "
           "mentioned explicitly");

    for (const auto &[Field, FieldInit] :
         llvm::zip(Def.getRecordType().getFields(), InitList->inits())) {
      const mlir::Location Loc = loc(FieldInit);
      const mlir::Value FieldMemory =
          getMemberPointer(Loc, InitializedValues.top(), Field);
      init(FieldMemory, FieldInit);
    }

    return {};
  }
  default:
    return VisitExpr(InitList);
  }
}

mlir::Value
FunctionGenerator::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *Literal) {
  return Builder.create<air::ConstantIntOp>(loc(Literal), Literal->getValue(),
                                            type(Literal).cast<IntegerType>());
}

//===----------------------------------------------------------------------===//
//                                Initialization
//===----------------------------------------------------------------------===//

void FunctionGenerator::init(mlir::Value Memory, const Expr *Init) {
  assert(Init != nullptr);

  InitializedValues.push(Memory);
  const auto RemoveInitializedValue = llvm::make_scope_exit([this, Memory]() {
    assert(!InitializedValues.empty() && InitializedValues.top() == Memory);
    InitializedValues.pop();
  });
  mlir::Value InitValue = Visit(Init);

  if (InitValue)
    store(InitValue, Memory, loc(Init));
}

//===----------------------------------------------------------------------===//
//                           Control flow statements
//===----------------------------------------------------------------------===//

mlir::Value FunctionGenerator::VisitIfStmt(const IfStmt *If) {
  DeclScope IfVariableScope(Declarations,
                            [](const ValueDecl *, mlir::Value) {});

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
  DeclScope WhileVariableScope(Declarations,
                               [](const ValueDecl *, mlir::Value) {});

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

  // Condition variables...
  if (While->getConditionVariableDeclStmt())
    Visit(While->getConditionVariableDeclStmt());

  // ...and conditions belong in the header.
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

mlir::Value FunctionGenerator::VisitForStmt(const ForStmt *ForLoop) {
  DeclScope ForVariableScope(Declarations,
                             [](const ValueDecl *, mlir::Value) {});

  // Here we deal with three basic blocks:
  //   * basic block for the header of the loop
  Block *Header = Target.addBlock();
  //   * basic block for the loop's body
  Block *Body = Target.addBlock();
  //   * basic block for the code after the loop
  Block *Tail = ForLoop->getInc() ? Target.addBlock() : Header;
  Block *Next = Target.addBlock();

  mlir::Location Loc = loc(ForLoop);

  // Loop variables can be even put into a separate "preheader" block,
  // but we are not pedantic on that part and simply put it into the
  // current block;
  if (ForLoop->getInit())
    Visit(ForLoop->getInit());

  // First, let's generate the loop's header.
  Builder.create<BranchOp>(Loc, Header);
  Builder.setInsertionPointToStart(Header);

  // Condition variables...
  if (ForLoop->getConditionVariableDeclStmt())
    Visit(ForLoop->getConditionVariableDeclStmt());

  // ... and conditions belong in the header.
  mlir::Value Cond = {};
  if (ForLoop->getCond())
    Cond = Visit(ForLoop->getCond());

  if (Cond)
    // If condition is true, we should proceed with the loop's
    // body, and skip it otherwise.
    Builder.create<air::CondBranchOp>(Loc, Cond, Body, Next);
  else
    // If we don't have a condition at all, jump to the body
    Builder.create<BranchOp>(Loc, Body);

  if (ForLoop->getInc()) {
    Builder.setInsertionPointToStart(Tail);
    Visit(ForLoop->getInc());
    Builder.create<BranchOp>(Loc, Header);
  }

  // Generate the body.
  Builder.setInsertionPointToStart(Body);
  Visit(ForLoop->getBody());

  // Let's check if the current block even needed.
  Block *Current = Builder.getBlock();
  if (Current->hasNoPredecessors())
    Current->erase();

  // And if yes, we might need to put a jump to the header block.
  else if (hasNoTerminator(Current)) {
    Builder.setInsertionPointToEnd(Current);
    Builder.create<BranchOp>(Loc, Tail);
  }

  // All further generation should happen here.
  Builder.setInsertionPointToStart(Next);

  return {};
}

mlir::Value FunctionGenerator::VisitExpr(const Expr *UnsupportedExpr) {
  const std::string Name = UnsupportedExpr->getStmtClassName();
  return Builder.create<air::UnsupportedOp>(loc(UnsupportedExpr),
                                            type(UnsupportedExpr), Name);
}
