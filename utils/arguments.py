"""Command line argument utility."""

from argparse   import _ArgumentGroup, ArgumentParser, Namespace, _SubParsersAction

# Initialize parser
_parser:            ArgumentParser =    ArgumentParser(
    prog =          "image-transform",
    description =   "Perform transformation on image (with wavelets)."
)

# Initialize subparsers
_subparser:         _SubParsersAction = _parser.add_subparsers(
    dest =          "cmd",
    help =          "Command being executed"
)

# +===============================================================================================+
# | BEGIN ARGUMENTS                                                                               |
# +===============================================================================================+

# LOGGING =========================================================================================
_logging:           _ArgumentGroup =    _parser.add_argument_group("Logging")

_logging.add_argument(
    "--logging_path",
    type =          str,
    default =       "logs",
    help =          "Path at which logs will be written. Defaults to \'./logs/\'."
)

_logging.add_argument(
    "--logging_level",
    type =          str,
    choices =       ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default =       "INFO",
    help =          "Minimum logging level (DEBUG < INFO < WARNING < ERROR < CRITICAL). Defaults to \'INFO\'."
)

# OUTPUT ==========================================================================================
_output:            _ArgumentGroup =    _parser.add_argument_group("Output")

_output.add_argument(
    "--output_path",
    type =          str,
    default =       "output",
    help =          "Path at which output files will be written. Defaults to \'./output/\'."
)

# CONVOLVE ========================================================================================
_convolve:          ArgumentParser =    _subparser.add_parser(
    name =          "convolve",
    help =          "Convolve over an image to observe the output"
)

_convolve.add_argument(
    "image",
    type =          str,
    help =          "Relative path to image being convolved."
)

# TRANSFORM =======================================================================================
_transform:         ArgumentParser =    _subparser.add_parser(
    name =          "wavelet-transform",
    help =          "Perform wavelet transform on an image"
)

_transform.add_argument(
    "image",
    type =          str,
    help =          "Relative path to image being transformed."
)

# WAVELETS ---------------------------------------------------------------
_wavelet_subparser: _SubParsersAction = _transform.add_subparsers(
    dest =          "wavelet",
    help =          "Wavelet being used in transform"
)

# MEXICAN-HAT ___________________________________
_mexican_hat:       ArgumentParser =    _wavelet_subparser.add_parser(
    name =          "mexican-hat",
    help =          "Mexican hat wavelet"
)

_mexican_hat.add_argument(
    "--amplitude", "-a",
    type =          int,
    default =       2,
    help =          "Amplitude of wavelet. Defaults to 2."
)

_mexican_hat.add_argument(
    "--center", "-cen",
    type =          float,
    default =       2.0,
    help =          "Center value of kernel. Scale of wavelet peak. Defaults to 2.0."
)

_mexican_hat.add_argument(
    "--edge", "-edg",
    type =          float,
    default =       -0.4,
    help =          "Edge value of kernel, Scale of wavelet's first outer wing. Defaults to -0.4."
)

_mexican_hat.add_argument(
    "--corner", "-cor",
    type =          float,
    default =       -0.1,
    help =          "Corner value of kernel. Scale of wavelet's second outer wing. Defaults to -0.1."
)

# +===============================================================================================+
# | END ARGUMENTS                                                                                 |
# +===============================================================================================+

# Parse command line arguments into namespace
ARGS:               Namespace = _parser.parse_args()