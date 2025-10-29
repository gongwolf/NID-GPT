import argparse
import pandas as pd
# from be_great import GReaT

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using a pretrained GReaT model."
    )

    # ---------------------------
    # Model loading
    # ---------------------------
    parser.add_argument(
        "--model_dir",
        type=str,
        default="CICIDS2017",
        help="Path to the directory of the fine-tuned GReaT model."
    )

    # ---------------------------
    # sample() parameters
    # ---------------------------
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4500,
        help="Number of synthetic samples to generate."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Sampling batch size. Higher values speed up the generation process."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2000,
        help="Maximum number of tokens to generate. Ensure it's long enough to not cut off any information."
    )
    parser.add_argument(
        "--start_col",
        type=str,
        default="label",
        help="Feature to use as the starting point for the generation process."
    )
    parser.add_argument(
        "--start_col_dist",
        type=str,
        default="{'0': 0.33, '1': 0.33, '2': 0.34}",
        help="Distribution of the starting feature (as a Python dictionary string)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for generation (e.g., 'cuda' or 'cpu')."
    )

    # ---------------------------
    # Output
    # ---------------------------
    parser.add_argument(
        "--output_csv",
        type=str,
        default="great_synthetic_data.csv",
        help="Path to save the generated synthetic data as CSV."
    )

    args = parser.parse_args()

    # Load the pretrained model
    model = GReaT.load_from_dir(args.model_dir)

    # Convert string dict input for start_col_dist into actual dict
    try:
        start_col_dist = eval(args.start_col_dist)
    except Exception:
        print("Warning: Could not parse start_col_dist. Using default distribution.")
        start_col_dist = {'0': 0.33, '1': 0.33, '2': 0.34}

    # Generate synthetic samples
    synthetic_data = model.sample(
        n_samples=args.n_samples,
        k=args.k,
        max_length=args.max_length,
        start_col=args.start_col,
        start_col_dist=start_col_dist,
        device=args.device
    )

    # Save generated data
    synthetic_data.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
