import argparse
from multiprocessing import cpu_count

def get_args():
    parser = argparse.ArgumentParser(
        description="SOAP-based structure selection tool.\n"
                    "Modes: 'fps' (Target Number) or 'threshold' (Cosine Similarity Cutoff).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files:
  train.xyz       : Selected structures
  test.xyz        : Remaining structures
  soap_cache/     : Cached descriptors
        """
    )

    parser.add_argument("-i", "--input", dest="input_path", type=str, default="total.xyz",
                        help="Input: single .xyz/.extxyz file or folder")

    parser.add_argument("-p", "--nproc", type=int, default=min(8, cpu_count()),
                        help="Number of processes")

    # === 新增：模式选择 ===
    parser.add_argument("-m", "--mode", type=str, default="fps", choices=["fps", "threshold"],
                        help="Selection mode: 'fps' (select fixed N) or 'threshold' (remove duplicates). Default: fps")

    # FPS 专用参数
    parser.add_argument("-n", "--num", type=int, default=1000,
                        help="[FPS Mode] Total number of structures to select. Default: 1000")

    # Threshold 专用参数
    parser.add_argument("-s", "--simlT", type=float, default=0.005,
                        help="[Threshold Mode] Similarity threshold (1 - cosine). Default: 0.005")

    parser.add_argument("-a", "--atoms", type=str, default="C H O",
                        help="SOAP centers (e.g. 'C H O')")

    parser.add_argument("-r", "--rcut", type=float, default=6.0,
                        help="SOAP cutoff radius")

    args = parser.parse_args()

    print("=" * 60)
    print(f"SOAP Selection Tool | Mode: {args.mode.upper()}")
    print(f"  Input           : {args.input_path}")
    if args.mode == "fps":
        print(f"  Target Total N  : {args.num}")
    else:
        print(f"  Siml Threshold  : {args.simlT} (1-Cos)")
    print(f"  Processes       : {args.nproc}")
    print("=" * 60)

    return args
