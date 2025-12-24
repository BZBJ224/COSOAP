import argparse
from multiprocessing import cpu_count

def get_args():
    parser = argparse.ArgumentParser(
        description="SOAP-based structure deduplication with labeled priority.\n"
                    "Supports single .xyz file or folder with multiple .xyz files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files:
  train.xyz       : Unique labeled structures
  test.xyz        : Similar labeled structures (for validation)
  unlabeled.xyz   : Deduplicated unlabeled structures
  uniques.csv     : Statistics per composition
  soap_cache/     : Cached descriptors (reused on rerun)
        """
    )

    parser.add_argument("-i", "--input", dest="input_path", type=str, default="total.xyz",
                        help="Input: single .xyz/.extxyz file or folder containing them")

    parser.add_argument("-p", "--nproc", type=int, default=min(8, cpu_count()),
                        help="Number of processes (default: auto)")

    parser.add_argument("-s", "--simlT", type=float, default=0.005,
                        help="Similarity threshold (1 - cosine), smaller = stricter (default: 0.005)")

    parser.add_argument("-a", "--atoms", type=str, default="C H O",
                        help="Space-separated elements used as SOAP centers (default: 'C H O')")

    parser.add_argument("-r", "--rcut", type=float, default=6.0,
                        help="SOAP cutoff radius in Å (default: 6.0)")

    args = parser.parse_args()

    print("=" * 60)
    print("SOAP Deduplication Parameters:")
    print(f"  Input           : {args.input_path}")
    print(f"  Processes       : {args.nproc}")
    print(f"  Similarity thresh: {args.simlT}")
    print(f"  SOAP centers    : {args.atoms}")
    print(f"  r_cut           : {args.rcut} Å")
    print("=" * 60)

    return args