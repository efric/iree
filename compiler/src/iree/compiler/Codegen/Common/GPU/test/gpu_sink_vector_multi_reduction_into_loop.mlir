// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-sink-vector-multi-reduction-into-loop))' | FileCheck %s

module @m {
  func.func @sink_basic(%A: vector<4x8xf32>, %B: vector<4x8xf32>, %acc: vector<4x8xf32>) -> vector<4xf32> {
    %c_start = arith.constant 0 : index
    %c_end = arith.constant 4 : index
    %c_step = arith.constant 1 : index
    %c_vector = arith.constant dense<0.000000e+00> : vector<4xf32>

    %result = scf.for %k = %c_start to %c_end step %c_step iter_args(%carry = %acc) -> (vector<4x8xf32>) {
      %mul = arith.mulf %A, %B : vector<4x8xf32>
      %add = arith.addf %carry, %mul : vector<4x8xf32>
      scf.yield %add : vector<4x8xf32>
    }

    %reduced = vector.multi_reduction <add>, %result, %c_vector [1] : vector<4x8xf32> to vector<4xf32>
    return %reduced : vector<4xf32>
  }
}