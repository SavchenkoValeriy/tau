//===- AirTypes.h - AIR own types -------------------------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  TBD
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace tau::air {

//===----------------------------------------------------------------------===//
//                                 Record type
//===----------------------------------------------------------------------===//

struct RecordField {
  llvm::StringRef Name;
  mlir::Type Type;
};

inline bool operator==(const RecordField &LHS, const RecordField &RHS) {
  return LHS.Name == RHS.Name && LHS.Type == RHS.Type;
}

// NOLINTNEXTLINE(readability-identifier-naming)
inline ::llvm::hash_code hash_value(tau::air::RecordField Field) {
  return hash_combine(hash_value(Field.Name), hash_value(Field.Type));
}

namespace detail {
struct RecordTypeStorage : public mlir::TypeStorage {
  using BasesTy = llvm::ArrayRef<mlir::Type>;
  using FieldsTy = llvm::ArrayRef<RecordField>;
  using KeyTy = std::pair<BasesTy, FieldsTy>;

  RecordTypeStorage(const BasesTy &Bases, const FieldsTy &Fields)
      : Bases(Bases), Fields(Fields) {}

  static RecordTypeStorage *construct(mlir::TypeStorageAllocator &Allocator,
                                      const KeyTy &Key) {
    // Copy both arrays into the allocator.
    const auto Bases = Allocator.copyInto(Key.first);
    const auto Fields = Allocator.copyInto(Key.second);

    return new (Allocator.allocate<RecordTypeStorage>())
        RecordTypeStorage(Bases, Fields);
  }

  bool operator==(const KeyTy &Key) const {
    return Key.first == Bases && Key.second == Fields;
  }

  BasesTy Bases;
  FieldsTy Fields;
};
} // end namespace detail

} // end namespace tau::air

#define GET_TYPEDEF_CLASSES
#include "tau/AIR/AirOpsTypes.h.inc"
