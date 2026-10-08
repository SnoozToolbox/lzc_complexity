import argparse
import ctypes
import json
import os

import numpy as np


def load_library(path):
    """Load the compiled LZC library with ctypes."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Library not found: {path}")

    print(f"Loading library: {path}")
    return ctypes.CDLL(os.path.abspath(path))


def call_complexity(lib, data):
    """Call the C complexity() function."""
    data = np.ascontiguousarray(data, dtype=np.float64)

    nc, nt = data.shape

    sig = (ctypes.c_double * nt)()
    out = ctypes.c_double()

    fn = lib.complexity

    fn.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        np.ctypeslib.ndpointer(
            dtype=np.float64,
            flags="C_CONTIGUOUS",
        ),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    fn.restype = None

    fn(
        nc,
        nt,
        data,
        2.0,
        sig,
        ctypes.byref(out),
    )

    return out.value


def call_complexity_joint(lib, data):
    """Call the C complexity_joint() function."""
    data = np.ascontiguousarray(data, dtype=np.float64)

    nc, nt = data.shape

    out = ctypes.c_double()

    fn = lib.complexity_joint

    fn.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        np.ctypeslib.ndpointer(
            dtype=np.float64,
            flags="C_CONTIGUOUS",
        ),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]

    fn.restype = None

    fn(
        nc,
        nt,
        data,
        2.0,
        ctypes.byref(out),
    )

    return out.value


def run_tests(lib):
    """
    Run deterministic test inputs through the native library.
    """

    # Test 1: simple 1D binary sequence
    test_1 = np.array([
        [0, 1, 0, 1, 1, 0, 1, 0,
         0, 1, 1, 1, 0, 0, 1, 0]
    ], dtype=np.float64)

    # Test 2: different 1D binary sequence
    test_2 = np.array([
        [0, 0, 1, 1, 0, 1, 1, 0,
         1, 0, 0, 1, 1, 1, 0, 1]
    ], dtype=np.float64)

    # Test 3: small multichannel binary matrix
    test_joint = np.array([
        [0, 1, 0, 1, 1, 0, 1, 0,
         0, 1, 1, 0, 1, 0, 1, 0],

        [1, 1, 0, 0, 1, 1, 0, 0,
         0, 1, 0, 1, 1, 0, 0, 1],

        [0, 0, 1, 1, 0, 0, 1, 1,
         1, 0, 1, 0, 0, 1, 1, 0],
    ], dtype=np.float64)

    results = {
        "complexity_test_1": call_complexity(lib, test_1),
        "complexity_test_2": call_complexity(lib, test_2),
        "complexity_joint_test": call_complexity_joint(lib, test_joint),
    }

    return results


def save_reference(results, output_file):
    """Save Windows results as the reference."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\nWindows reference results:")
    for name, value in results.items():
        print(f"  {name}: {value}")

    print(f"\nReference saved to: {output_file}")


def compare_results(results, reference_file):
    """Compare this OS results with the Windows reference."""

    with open(reference_file, "r") as f:
        reference = json.load(f)

    print("\nReference results:")
    for name, value in reference.items():
        print(f"  {name}: {value}")

    print("\nCurrent OS results:")
    for name, value in results.items():
        print(f"  {name}: {value}")

    print("\nComparison:")

    all_passed = True

    for name in reference:

        expected = reference[name]
        actual = results[name]

        if np.isclose(actual, expected):
            print(
                f"  PASS: {name} "
                f"(reference={expected}, current={actual})"
            )
        else:
            print(
                f"  FAIL: {name} "
                f"(reference={expected}, current={actual})"
            )
            all_passed = False

    if not all_passed:
        raise AssertionError(
            "The native library results do not match "
            "the Windows reference."
        )

    print("\nAll results match the Windows reference!")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--library",
        required=True,
        help="Path to the compiled native library",
    )

    parser.add_argument(
        "--save-reference",
        help="Save results to a JSON reference file",
    )

    parser.add_argument(
        "--compare",
        help="Compare results with a JSON reference file",
    )

    args = parser.parse_args()

    lib = load_library(args.library)

    results = run_tests(lib)

    if args.save_reference:
        save_reference(
            results,
            args.save_reference,
        )

    elif args.compare:
        compare_results(
            results,
            args.compare,
        )

    else:
        print(results)


if __name__ == "__main__":
    main()