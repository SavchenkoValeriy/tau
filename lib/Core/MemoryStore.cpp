#include "tau/Core/MemoryStore.h"

#include "tau/AIR/AirOps.h"
#include "tau/Support/FunctionExtras.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <immer/map_transient.hpp>
#include <immer/set_transient.hpp>
#include <variant>

using namespace tau;
using namespace air;
using namespace core;

//===----------------------------------------------------------------------===//
//                                  Utilities
//===----------------------------------------------------------------------===//

namespace {
struct PointsTo {};
struct Field {
  llvm::StringRef Name;
};
struct Element {
  unsigned Index;
};

/// Represents the various ways one value can be related to another in memory
using Relationship = std::variant<PointsTo, Field, Element>;

bool isRelationshipEqual(const Relationship &LHS, const Relationship &RHS) {
  if (std::holds_alternative<PointsTo>(LHS) &&
      std::holds_alternative<PointsTo>(RHS))
    return true;

  if (const Field *LField = std::get_if<Field>(&LHS),
      *RField = std::get_if<Field>(&RHS);
      LField != nullptr && RField != nullptr) {
    return LField->Name == RField->Name;
  }

  if (const Element *LElement = std::get_if<Element>(&LHS),
      *RElement = std::get_if<Element>(&RHS);
      LElement != nullptr && RElement != nullptr) {
    return LElement->Index == RElement->Index;
  }

  return false;
}

[[maybe_unused]] llvm::StringRef getName(Relationship Rel) {
  if (std::holds_alternative<Field>(Rel)) {
    return "Field";
  }
  if (std::holds_alternative<Element>(Rel)) {
    return "Element";
  }
  return "PointsTo";
}

} // end anonymous namespace

struct MemoryStore::MemoryKey {
  mlir::Value Value;
  Relationship Rel;

  bool operator==(const MemoryKey &Other) const {
    return Value == Other.Value && isRelationshipEqual(Rel, Other.Rel);
  }
};

namespace std {
template <> struct hash<tau::core::MemoryStore::MemoryKey> {
  size_t operator()(const tau::core::MemoryStore::MemoryKey &Key) const {
    static const auto PointsToHash = llvm::hash_value("##points-to##");
    const auto VariantHash = std::visit(
        tau::support::Overloaded{
            [](PointsTo) { return PointsToHash; },
            [](Field Field) { return llvm::hash_value(Field.Name); },
            [](Element Element) { return llvm::hash_value(Element.Index); }},
        Key.Rel);
    return llvm::hash_combine(mlir::hash_value(Key.Value), VariantHash);
  }
};

template <> struct hash<mlir::Value> {
  size_t operator()(const mlir::Value &Value) const {
    return mlir::hash_value(Value);
  }
};

template <> struct hash<tau::core::MemoryStore::Definition> {
  size_t operator()(const tau::core::MemoryStore::Definition &Def) const {
    return llvm::hash_combine(Def.Value,
                              Def.Event ? Def.Event->getLocation() : nullptr);
  }
};

} // end namespace std

//===----------------------------------------------------------------------===//
//                                   Builder
//===----------------------------------------------------------------------===//

/// Builder class for constructing updated MemoryStore instances.
/// This allows for efficient updates to the immutable MemoryStore structure.
class MemoryStore::Builder {
public:
  Builder(MemoryStore Base)
      : BaseStore(Base), Model(Base.Model.transient()),
        Canonicals(Base.Canonicals.transient()) {}

  MemoryStore build() {
    return MemoryStore(BaseStore.Hierarchy, Model.persistent(),
                       Canonicals.persistent());
  }

  void associate(mlir::Value Base, Relationship Rel, mlir::Value Result) {
    const SetOfValues BaseCanonicals = BaseStore.getDefininingValues(Base);
    SetOfValues::transient_type AlreadyAssociated;
    for (Definition KnownBase : BaseCanonicals) {
      Model.update(MemoryKey{KnownBase.Value, Rel},
                   [Result, &AlreadyAssociated](const SetOfValues &Current) {
                     if (Current.empty())
                       return SetOfValues{{Result}};

                     for (Definition Associated : Current) {
                       AlreadyAssociated.insert(Associated);
                     }

                     return Current;
                   });
    }

    setCanonicals(Result, AlreadyAssociated.persistent());
  }

  void store(mlir::Value Base, Relationship Rel, mlir::Value ValueToStore,
             mlir::Operation *Op) {
    const SetOfValues BaseCanonicals = BaseStore.getDefininingValues(Base);
    for (Definition KnownBase : BaseCanonicals) {
      Model.set(MemoryKey{KnownBase.Value, Rel},
                addStoreEvent(Op, BaseStore.getDefininingValues(ValueToStore)));
    }
  }

  void alias(mlir::Value From, mlir::Value To) {
    setCanonicals(From, BaseStore.getDefininingValues(To));
  }

  void setCanonical(mlir::Value For, mlir::Value Canonical) {
    if (For != Canonical)
      setCanonicals(For, SetOfValues{{Canonical}});
  }

  void setCanonicals(mlir::Value For, SetOfValues NewCanonicals) {
    if (!NewCanonicals.empty())
      Canonicals.set(For, NewCanonicals);
  }

  SetOfValues addStoreEvent(mlir::Operation *StoreOp,
                            SetOfValues StoredValues) {
    SetOfValues::transient_type Result;

    for (const Definition &Def : StoredValues) {
      const DataFlowEvent *NewEvent = nullptr;
      if (Def.Event == nullptr) {
        NewEvent = &BaseStore.Hierarchy.get().addDataFlowEvent(StoreOp);
      } else {
        NewEvent =
            &BaseStore.Hierarchy.get().addDataFlowEvent(StoreOp, {Def.Event});
      }
      Result.insert({Def.Value, NewEvent});
    }

    return Result.persistent();
  }

  void join(MemoryStore Other) {
    const auto MakeSetUpdater = [](SetOfValues With) {
      return [With](const SetOfValues &Current) {
        if (Current.empty())
          return With;

        auto TransientCurrent = Current.transient();
        for (Definition OtherCanonical : With) {
          TransientCurrent.insert(OtherCanonical);
        }
        return TransientCurrent.persistent();
      };
    };

    for (auto &[From, To] : Other.Canonicals) {
      Canonicals.update(From, MakeSetUpdater(To));
    }

    for (auto &[Key, Values] : Other.Model) {
      Model.update(Key, MakeSetUpdater(Values));
    }
  }

private:
  MemoryStore BaseStore;
  ModelTy::transient_type Model;
  CanonicalsTy::transient_type Canonicals;
};

//===----------------------------------------------------------------------===//
//                       Public interface implementation
//===----------------------------------------------------------------------===//

MemoryStore::MemoryStore(EventHierarchy &Hierarchy) : Hierarchy(Hierarchy) {}

MemoryStore::~MemoryStore() = default;

MemoryStore::MemoryStore(const MemoryStore &) = default;
MemoryStore &MemoryStore::operator=(const MemoryStore &) = default;

MemoryStore::MemoryStore(MemoryStore &&) = default;
MemoryStore &MemoryStore::operator=(MemoryStore &&) = default;

MemoryStore::MemoryStore(EventHierarchy &Hierarchy, MemoryStore::ModelTy Model,
                         MemoryStore::CanonicalsTy Canonicals)
    : Hierarchy(Hierarchy), Model(Model), Canonicals(Canonicals) {}

MemoryStore MemoryStore::interpret(mlir::Operation *Op) {
  Builder B(*this);

  mlir::TypeSwitch<mlir::Operation *, void> Switch(Op);
  Switch
      .Case<air::LoadOp>([&B](air::LoadOp Load) {
        B.associate(Load.getAddress(), PointsTo{}, Load.getResult());
      })
      .Case<air::StoreOp>([&B, Op](air::StoreOp Store) {
        B.store(Store.getAddress(), PointsTo{}, Store.getValue(), Op);
      })
      .Case<air::GetFieldPtr>([&B](air::GetFieldPtr FieldPtr) {
        B.associate(FieldPtr.getRecord(), Field{FieldPtr.getFieldAttrName()},
                    FieldPtr.getRes());
      })
      .Case<air::NoOp>(
          [&B](air::NoOp Noop) { B.alias(Noop.getRes(), Noop.getValue()); })
      .Case<air::RefOp>(
          [&B](air::RefOp Ref) { B.alias(Ref.getRes(), Ref.getValue()); });
  return B.build();
}

MemoryStore::SetOfValues
MemoryStore::getDefininingValues(mlir::Value Value) const {
  if (const auto *DefinitingValues = Canonicals.find(Value)) {
    return *DefinitingValues;
  }
  return {{Value}};
}

MemoryStore MemoryStore::join(MemoryStore Other) {
  Builder B(*this);
  B.join(Other);
  return B.build();
}

bool MemoryStore::operator==(const MemoryStore &Other) const {
  return Canonicals == Other.Canonicals && Model == Other.Model;
}

bool MemoryStore::operator!=(const MemoryStore &Other) const {
  return !(*this == Other);
}
