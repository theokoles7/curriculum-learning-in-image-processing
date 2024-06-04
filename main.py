"""Drive application."""

from os         import makedirs

from command    import convolve, transform
from utils      import ARGS, LOGGER
from wavelets   import MexicanHatWavelet

if __name__ == "__main__":
    """Execute application."""
    
    try:
        # Log banner
        LOGGER.info("+==============================================================================================================+")
        LOGGER.info("| DADL Lab: Image Complexity Analysis | Copright 2024 Ashton Andrepont, Gabriel C. Trahan, Dr. MD Aminul Islam |")
        LOGGER.info("+==============================================================================================================+")
        
        # Ensure output path exists
        makedirs(ARGS.output_path, exist_ok = True)
        
        # Match command
        match ARGS.cmd:
            
            # CONVOLUTION =========================================================================
            case "convolve":
                
                # Log action
                LOGGER.info(f"Performing convolution on image: {ARGS.image}")
                
                convolve(ARGS.image, ARGS.kernel)
            
            # WAVELET TRANSFORM ===================================================================
            case "wavelet-transform":
                
                # Log action
                LOGGER.info(f"Performing {ARGS.wavelet} wavelet transform on image: {ARGS.image}")
                
                # Match wavelet
                match ARGS.wavelet:
                    
                    # MEXICAN HAT ----------------------------------------
                    case "mexican-hat":
                        
                        # Initialize Mexican-Hat wavelet
                        wavelet:    MexicanHatWavelet = MexicanHatWavelet(ARGS.center, ARGS.edge, ARGS.corner, ARGS.amplitude)._kernel
                      
                    # IMPROPER WAVELET SELECTION -------------------------  
                    case _:
                        raise NotImplementedError(f"Wavelet transform not yet implemented for {ARGS.wavelet}")
                    
                # Perform transform
                transform(ARGS.image, wavelet)
               
            # IMPROPER COMMAND ====================================================================
            case _:
                raise NotImplementedError(f"Command provided has not been implemented: {ARGS.cmd}")
        
    # Gracefully handle interrupts
    except KeyboardInterrupt:
        
        LOGGER.critical("Keyboard interrupt detected...")
        
    # Handle other erros and record stacktrace for investigation
    except Exception as e:
        
        LOGGER.error(f"An error occured: {e}", exc_info = True)
        
    # Graceful exit
    finally:
        
        LOGGER.info("Exiting...")