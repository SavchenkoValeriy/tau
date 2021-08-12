include "mlir/IR/OpBase.td"


def AIRDialect : Dialect {
    let name = "air";
    let summary = "Analysis-purposed MLIR dialect.";
    let description = [{
        
    }];
    let cppNamespace = "air";
}
