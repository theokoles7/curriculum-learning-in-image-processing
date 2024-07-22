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
                git add ./output/results.csv
                git commit -m "$(date +'%F %T'): $model | $dataset | $epochs | $batch_size | control"
                git push origin main

                # For each curriculum...
                for curriculum in rmse, spatial_frequency, wavelet_energy, wavelet_entropy
                do

                    # Run by entire dataset
                    python -m main run-job          \
                        --epochs        $epochs     \
                        $model                      \
                        $dataset                    \
                        --batch_size    $batch_size \
                        --curriculum    $curriculum

                    # Push updated results
                    git add ./output/results.csv
                    git commit -m "$(date +'%F %T'): $model | $dataset | $epochs | $batch_size | $curriculum"
                    git push origin main

                    # Run by batch
                    python -m main run-job          \
                        --epochs        $epochs     \
                        $model                      \
                        $dataset                    \
                        --batch_size    $batch_size \
                        --curriculum    $curriculum \
                        --by_batch

                    # Push updated results
                    git add ./output/results.csv
                    git commit -m "$(date +'%F %T'): $model | $dataset | $epochs | $batch_size | $curriculum | by-batch"
                    git push origin main

                done

            done

        done

    done

done