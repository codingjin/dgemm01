import numpy as np
import tvm
from tvm import te, auto_scheduler

import sys

@auto_scheduler.register_workload
def matmul_add(M, N, K, dtype):
    A = te.placeholder((M, K), name = "A", dtype = dtype)
    B = te.placeholder((K, N), name = "B", dtype = dtype)
    C = te.placeholder((M, N), name = "C", dtype = dtype)

    k = te.reduce_axis((0, K), name = "k")
    matmul = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis = k),
        name = "matmul",
        attrs = {"layout_free_placeholders": [B]},
    )

    out = te.compute((M, N), lambda i, j: matmul[i, j] + C[i, j], name = "out")

    return [A, B, C, out]

def main():
    argv = sys.argv
    #print(len(argv))
    if len(argv) != 4:
        print("Invalid input!")
        print("python autotvm_mm02.py M N K!")

    M = int(argv[1])
    N = int(argv[2])
    K = int(argv[3])
    target = tvm.target.Target("llvm")
    task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(M, N, K, "float32"), target=target)

    #print("Computational DAG:")
    #print(task.compute_dag)

    log_file = "matmul.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials = 1000,
        measure_callbacks = [auto_scheduler.RecordToFile(log_file)],
        verbose = 0,
    )

    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)


    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    c_np = np.random.uniform(size=(M, N)).astype(np.float32)
    out_np = a_np.dot(b_np) + c_np

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)
    func(a_tvm, b_tvm, c_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, number = 5)
    #evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    print(
        "M=%d N=%d K=%d    Performance: %.3f GFLOPs/s" 
        % (M, N, K, 2.0e-9*M*N*K/np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results))
    )
    #print(
    #    "Execution time of this operator: %.3f ms"
    #    % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
    #)


if __name__ == "__main__":
    main()


