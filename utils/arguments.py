"""Command line argument parsing & utilities."""

from argparse   import ArgumentParser, _ArgumentGroup, Namespace, _SubParsersAction

# Intialize parser
__parser:           ArgumentParser =    ArgumentParser(
    prog =          "iccl",
    description =   "Image complexity metrics analysis."
)

__sub_parser:       _SubParsersAction = __parser.add_subparsers(
    dest =          "cmd",
    help =          "Command being executed."
)

# +================================================================================================+
# | BEGIN ARGUMENTS                                                                                |
# +================================================================================================+

# LOGGING ==========================================================================================
__logging:          _ArgumentGroup =    __parser.add_argument_group("Logging")

__logging.add_argument(
    "--logging_path",
    type =          str,
    default =       "logs",
    help =          "Path at which logs will be written. Defaults to \'./logs/\'."
)

__logging.add_argument(
    "--logging_level",
    type =          str,
    choices =       ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default =       "INFO",
    help =          "Minimum logging level (DEBUG < INFO < WARNING < ERROR < CRITICAL). Defaults to \'INFO\'."
)

# OUTPUT ==========================================================================================
__output:           _ArgumentGroup =    __parser.add_argument_group("Output")

__output.add_argument(
    "--output_path",
    type =          str,
    default =       "output",
    help =          "Path at which output files/reports will be written. Defaults to \'./output/\'."
)

# INITIALIZE RESULTS ===============================================================================
__init_results:     ArgumentParser =    __sub_parser.add_parser(
    name =          "init-results",
    help =          "Initialize results file."
)

# RUN JOB ==========================================================================================
__job:              ArgumentParser =    __sub_parser.add_parser(
    name =          "run-job",
    help =          "Execute job."
)

__job.add_argument(
    "--epochs",
    type =          int,
    default =       20,
    help =          "Number of epochs for which job should run. Defaults to 20."
)

# MODEL ------------------------------------------------------------------
__model:            _ArgumentGroup =    __job.add_argument_group("Model")

__model.add_argument(
    "model",
    type =          str,
    choices =       ["cnn"],
    default =       "cnn",
    help =          "Model selection. Defaults to \'cnn\'."
)

# OPTIMIZER --------------------------------------------------------------
__optimizer:        _ArgumentGroup =    __job.add_argument_group("Optimizer")

__optimizer.add_argument(
    "--optimizer",
    type =          str,
    choices =       ["adam", "sgd"],
    default =       "adam",
    help =          "Optimizer selection. Defaults to Adam."
)

__optimizer.add_argument(
    "--learning_rate", "-lr",
    type =          float,
    default =       0.01,
    help =          "Optimizer learning rate. Defaults to 0.01."
)

# DATASET ----------------------------------------------------------------
__dataset:          _ArgumentGroup =    __job.add_argument_group("Dataset")

__dataset.add_argument(
    "dataset",
    type =          str,
    choices =       ["cifar10", "cifar100"],
    default =       "cifar10",
    help =          "Dataset selection. Defaults to \'cifar10\'."
)

__dataset.add_argument(
    "--dataset_path",
    type =          str,
    default =       "data",
    help =          "Path at which datasets can be located/downloaded. Defaults to \'./data/\'."
)

__dataset.add_argument(
    "--batch_size", "-bs",
    type =          int,
    default =       16,
    help =          "Dataset batch size. Defaults to 16."
)

# CURRICULUM -------------------------------------------------------------
__curriculum:       _ArgumentGroup =    __job.add_argument_group("Curriculum")

__curriculum.add_argument(
    "--curriculum",
    type =          str,
    choices =       ["edge_density", "rmse", "spatial_frequency", "wavelet_energy", "wavelet_entropy", "none"],
    default =       None,
    help =          "Curriculum selection. Defaults to None."
)

__curriculum.add_argument(
    "--by_batch",
    action =        "store_true",
    default =       False,
    help =          "Sort individual batches, instead of entire dataset. Defaults to False."
)

# RUN EXPERIENT ====================================================================================
__experiment:       ArgumentParser =    __sub_parser.add_parser(
    name =          "run-experiment",
    help =          "Execute experiment set."
)

# +================================================================================================+
# | END ARGUMENTS                                                                                  |
# +================================================================================================+

# Parse command line arguments into name space
ARGS:               Namespace =         __parser.parse_args()