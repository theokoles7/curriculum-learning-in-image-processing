"""Initialize results.csv file."""

from csv    import writer
from os     import makedirs

from utils  import ARGS

def init_results() -> None:
    """Initialize results CSV file."""
    
    # Ensure output directory exists
    makedirs(ARGS.output_path, exist_ok = True)
    
    # Create/open CSV file
    with open(f"{ARGS.output_path}/results.csv", "w", newline = "", encoding = "utf-8") as file_out:
        
        print("Wiriting results.csv")
        
        report: writer =    writer(file_out)
        
        report.writerow(["MODEL", "DATASET", "CURRICULUM", "BY BATCH", "EPOCHS", "BATCH SIZE", "AVG TRAIN ACCURACY", "AVG TRAIN LOSS", "AVG VALIDATION ACCURACY", "AVG VALIDATION LOSS", "TEST ACCURACY", "TEST LOSS"])
        
        # Write rows
        for model in ["cnn"]:
            for dataset in ["cifar10", "cifar100"]:
                for curriculum in ["none", "rmse", "spatial_frequency", "wavelet_energy", "wavelet_entropy"]:
                    for epochs in ["50", "100", "200"]:
                        for batch_size in ["8", "16", "32", "64"]:
                            
                            if curriculum != "none":
                                for by_batch in [True, False]:
                                    
                                    print(f"Writing {model} | {dataset} | {curriculum} | {by_batch} | {epochs} | {batch_size}")
                                    
                                    # Write row
                                    report.writerow([
                                        model,
                                        dataset,
                                        curriculum,
                                        by_batch,
                                        epochs,
                                        batch_size,
                                        "--", "--", "--", "--", "--", "--"
                                    ])
                            
                            else:
                                    
                                print(f"Writing {model} | {dataset} | {curriculum} | -- | {epochs} | {batch_size}")
                                    
                                # Write row
                                report.writerow([
                                    model,
                                    dataset,
                                    curriculum,
                                    False,
                                    epochs,
                                    batch_size,
                                    "--", "--", "--", "--", "--", "--"
                                ])