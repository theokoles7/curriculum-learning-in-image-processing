"""Drive application."""

from commands   import init_results, Job
from utils  import ARGS, BANNER, LOGGER

if __name__ == "__main__":
    """Execute commands."""

    try:

        LOGGER.info(BANNER)

        # Match command
        match ARGS.cmd:
            
            case "init-results":    init_results()

            case "run-job":         Job().run()
            
            case _:                 raise NotImplementedError(f"{ARGS.cmd} is not a valid command.")

    except KeyboardInterrupt:

        LOGGER.warning("Keyboard interrupt detected...")

    except Exception as e:

        LOGGER.error(f"An error occurred: {e}\n", exc_info = True)

    finally:

        LOGGER.info("Exiting...")