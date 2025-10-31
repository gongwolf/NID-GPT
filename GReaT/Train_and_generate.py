import argparse
import pandas as pd
from be_great import GReaT

def main():
    parser = argparse.ArgumentParser(
        description="Train and sample synthetic data using the GReaT generative model."
    )

    # ---------------------------
    # GReaT() parameters
    # ---------------------------
    parser.add_argument(
        "--llm",
        type=str,
        default="EleutherAI/gpt-neo-125m",
        help="HuggingFace checkpoint of a pretrained large language model used as the base model."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="trained_models",
        help="Directory where the training checkpoints will be saved."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size used for fine-tuning."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to fine-tune the model."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save the model after this many steps."
    )

    # ---------------------------
    # sample() parameters
    # ---------------------------
    parser.add_argument(
        "--n_samples",
        type=int,
        default=25000,
        help="Number of synthetic samples to generate."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
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
    # parser.add_argument(
    #     "--start_col_dist",
    #     type=str,
    #     default="{'0': 0.06, '1': 0.06, '2': 0.2, '3': 0.2, '4': 0.2, '5': 0.2, '6': 0.06, '7': 0.02}",
    #     help="Distribution of the starting feature (as a Python dictionary string)."
    # )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for generation (e.g., 'cuda' or 'cpu')."
    )

    # ---------------------------
    # I/O paths
    # ---------------------------
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/CICIDS2017/train_all_classes.csv",
        # choices = ["../data/CICIDS2017/train_all_classes.csv", "../data/CICDDOS2019/training.csv", "../data/UNSW/train_all_classes.csv"],
        help="Path to the training CSV data."
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="final_model",
        help="Path to save the trained model."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="great_synthetic_data.csv",
        help="Path to save the generated synthetic data as CSV."
    )

    args = parser.parse_args()

    # Load training data
    data = pd.read_csv(args.data_path)

    # Initialize GReaT model
    model = GReaT(
        llm=args.llm,
        experiment_dir=args.experiment_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_steps=args.save_steps
    )

    # Train the model
    model.fit(data)

    # Save the trained model
    model.save(args.save_model_path)

    # Convert string dict input for start_col_dist into actual dict
    # try:
    #     start_col_dist = eval(args.start_col_dist)
    # except Exception:
    #     print("Warning: Could not parse start_col_dist. Using default distribution.")
    #     start_col_dist = {'0': 0.06, '1': 0.06, '2': 0.2, '3': 0.2, '4': 0.2, '5': 0.2, '6': 0.06, '7': 0.02}

    # Generate synthetic samples
    synthetic_data = model.sample(
        n_samples=args.n_samples,
        k=args.k,
        max_length=args.max_length,
        start_col=args.start_col,
        # start_col_dist=start_col_dist,
        device=args.device
    )

    # Save generated data
    synthetic_data.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
