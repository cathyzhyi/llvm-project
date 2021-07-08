// RUN: mlir-opt %s -linalg-bufferize -std-bufferize \
// RUN: -tensor-constant-bufferize -tensor-bufferize -func-bufferize \
// RUN: -finalizing-bufferize \
// RUN: -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %const = constant dense<[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]> : tensor<1x2x3xf32>
  %dynamic = tensor.cast %const: tensor<1x2x3xf32> to tensor<1x?x3xf32>
  %offset = constant 2 : index
  %padded = call @pad_tensor(%dynamic, %offset) : (tensor<1x?x3xf32>, index) -> (tensor<?x?x?xf32>)
  %unranked = tensor.cast %padded: tensor<?x?x?xf32> to tensor<*xf32>
  call @print_memref_f32(%unranked) : (tensor<*xf32>) -> ()
  return
}

func @pad_tensor(%arg0: tensor<1x?x3xf32>, %arg1: index) -> tensor<?x?x?xf32> {
  %cst = constant 2.3 : f32
  %c0 = constant 0 : index
  %out = linalg.pad_tensor %arg0 low[%c0, %arg1, %c0] high[%c0, %c0, %arg1]  {
  ^bb0(%gen_arg1: index, %gen_arg2: index, %gen_arg3: index):  // no predecessors
    linalg.yield %cst : f32
  } : tensor<1x?x3xf32> to tensor<1x?x?xf32>
  %dynamic = tensor.cast %out: tensor<1x?x?xf32> to tensor<?x?x?xf32>
  return %dynamic: tensor<?x?x?xf32>
}

func private @print_memref_f32(%ptr : tensor<*xf32>)
