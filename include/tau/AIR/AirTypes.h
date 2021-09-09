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

#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

namespace tau::air {

namespace detail {
struct PointerTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  PointerTypeStorage(const KeyTy &Key) : PointeeType(Key) {}

  static PointerTypeStorage *construct(mlir::TypeStorageAllocator &Allocator,
                                       const KeyTy &Key) {
    return new (Allocator.allocate<PointerTypeStorage>())
        PointerTypeStorage(Key);
  }

  bool operator==(const KeyTy &Key) const { return Key == PointeeType; }

  mlir::Type PointeeType;
};
} // end namespace detail

//===----------------------------------------------------------------------===//
//                                 Pointer type
//===----------------------------------------------------------------------===//

class PointerType : public mlir::Type::TypeBase<PointerType, mlir::Type,
                                                detail::PointerTypeStorage> {
public:
  /// Inherit base constructors.
  using Base::Base;
  using Base::getChecked;

  /// Gets or creates an instance of LLVM dialect pointer type pointing to an
  /// object of `Pointee` type in the given address space. The pointer type is
  /// created in the same context as `Pointee`.
  static PointerType get(mlir::Type Pointee);

  /// Returns the pointed-to type.
  mlir::Type getElementType() const;
};
} // end namespace tau::air
