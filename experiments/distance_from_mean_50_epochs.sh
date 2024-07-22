#!/bin/bash

############################################
# RUN JOBS ON ALL VARIATIONS OF PARAMETERS #
############################################

# For each model...
for model in cnn
do

    # dataset...
    for dataset in cifar10 cifar100
    do

        # number of epochs...
        for epochs in 50
        do

            # and batch_size...
            for batch_size in 16
            do

                # Run control
                python -m main run-job          \
                    --epochs        $epochs     \
                    $model                      \
                    $dataset                    \
                    --batch_size    $batch_size

                # Push updated results
                git add ./output/distance_from_mean_results.csv
                git commit -m "$(date +'%F %T'): $model | $dataset | $epochs | $batch_size | control | from mean"
                git push origin Distance_From_Mean

                # For each curriculum...
                for curriculum in rmse, spatial_frequency, wavelet_energy, wavelet_entropy
                do

                    # Run by entire dataset
                    python -m main run-job          \
                        --epochs        $epochs     \
                        $model                      \
                        $dataset                    \
                        --batch_size    $batch_size \
                        --curriculum    $curriculum \
                        --sort_mean


                    # Push updated results
                    git add ./output/distance_from_mean_results.csv
                    git commit -m "$(date +'%F %T'): $model | $dataset | $epochs | $batch_size | $curriculum | from mean"
                    git push origin Distance_From_Mean

                    # Run by batch
                    python -m main run-job          \
                        --epochs        $epochs     \
                        $model                      \
                        $dataset                    \
                        --batch_size    $batch_size \
                        --curriculum    $curriculum \
                        --by_batch                  \
                        --sort_mean

                    # Push updated results
                    git add ./output/distance_from_mean_results.csv
                    git commit -m "$(date +'%F %T'): $model | $dataset | $epochs | $batch_size | $curriculum | by-batch | from mean"
                    git push origin Distance_From_Mean

                done

            done

        done

    done

done