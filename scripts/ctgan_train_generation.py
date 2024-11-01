#%%
import pandas as pd 
import argparse

from sdv.metadata import Metadata
import os
from sdv.single_table import CTGANSynthesizer

def load_trained_model(model_path, cuda=True):
    """
    Loads a trained model from a file path.

    Parameters:
    - model_path (str): Path to the trained model file.
    - cuda (bool): Whether to enable GPU support.

    Returns:
    - synthesizer (CTGANSynthesizer): The loaded CTGAN model.
    """
    synthesizer = CTGANSynthesizer.load(filepath=model_path)
    synthesizer.cuda = cuda
    return synthesizer

def main(data_file, ctgan_metadata_path, ctgan_model_save_path, ctgan_data_save_path, epochs, num_rows, use_cuda, table_name, sample_only:bool):
    # Load data


    # Prepare Metadata
    if not sample_only: #if only need the sample process, don't need to delete the metafile first
        if os.path.exists(ctgan_metadata_path):
            os.remove(ctgan_metadata_path)
            print(f"File '{ctgan_metadata_path}' deleted successfully.")
        df = pd.read_csv(data_file)
        print(f"Data loaded with shape: {df.shape}")
        metadata = Metadata.detect_from_dataframe(df, table_name=table_name)
        metadata.save_to_json(filepath=ctgan_metadata_path)
        synthesizer = CTGANSynthesizer(metadata, epochs=epochs, verbose=True, cuda=use_cuda)
        synthesizer.fit(df)
        synthesizer.save(filepath=ctgan_model_save_path)
    else:
        metadata = Metadata.load_from_json(filepath=ctgan_metadata_path)
        synthesizer=load_trained_model(model_path=ctgan_model_save_path, cuda  = use_cuda)

    # Train CTGAN Synthesizer


    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    synthetic_data.to_csv(ctgan_data_save_path, index=False)

    # Save the trained synthesizer and synthetic data
    print(f"Synthetic data saved to {ctgan_data_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a CTGAN model with synthetic data generation.")

    # Add arguments for the configurable options
    parser.add_argument('--input', type=str, help="Path to the input data file (CSV format).")
    parser.add_argument('--metadata', type=str, required=True, help="Path to save the CTGAN metadata JSON file.")
    parser.add_argument('--model_path', type=str, help="Path to save the trained CTGAN model file.")
    parser.add_argument('--output', type=str, help="Path to save the generated synthetic data (CSV format).")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs for CTGAN.")
    parser.add_argument('--num_rows', type=int, default=5000, help="Number of synthetic rows to generate.")
    parser.add_argument('--use_cuda', action='store_true', help="Flag to enable CUDA (GPU) support if available.")
    parser.add_argument('--table_name', type=str, default='ctgan_data', help="the table name showed in the metafile")
    parser.add_argument("--sample_only", action='store_true',  default=False, help="If only do the samppling")

    args = parser.parse_args()

    print(args)

    # Call the main function with parsed arguments
    main(
        data_file=args.input,
        ctgan_metadata_path=args.metadata,
        ctgan_model_save_path=args.model_path,
        ctgan_data_save_path=args.output,
        epochs=args.epochs,
        num_rows=args.num_rows,
        use_cuda=args.use_cuda,
        table_name = args.table_name,
        sample_only = args.sample_only
    )