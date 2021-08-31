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

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace tau::air {

namespace detail {
struct AirPointerTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  AirPointerTypeStorage(const KeyTy &Key) : PointeeType(Key) {}

  static AirPointerTypeStorage *construct(mlir::TypeStorageAllocator &Allocator,
                                          const KeyTy &Key) {
    return new (Allocator.allocate<AirPointerTypeStorage>())
        AirPointerTypeStorage(Key);
  }

  bool operator==(const KeyTy &Key) const { return Key == PointeeType; }

  mlir::Type PointeeType;
};
} // end namespace detail

//===----------------------------------------------------------------------===//
//                                 Pointer type
//===----------------------------------------------------------------------===//

class AirPointerType
    : public mlir::Type::TypeBase<AirPointerType, mlir::Type,
                                  detail::AirPointerTypeStorage> {
public:
  /// Inherit base constructors.
  using Base::Base;
  using Base::getChecked;

  /// Gets or creates an instance of LLVM dialect pointer type pointing to an
  /// object of `Pointee` type in the given address space. The pointer type is
  /// created in the same context as `Pointee`.
  static AirPointerType get(mlir::Type Pointee);

  /// Returns the pointed-to type.
  mlir::Type getElementType() const;
};
} // end namespace tau::air
