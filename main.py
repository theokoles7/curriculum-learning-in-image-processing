"""Drive application."""

from commands   import Job
from utils  import ARGS, BANNER, LOGGER

if __name__ == "__main__":
    """Execute commands."""

    try:

        LOGGER.info(BANNER)

        # Match command
        match ARGS.cmd:

            case "run-job":

                Job().run()

    except KeyboardInterrupt:

        LOGGER.warning("Keyboard interrupt detected...")

    except Exception as e:

        LOGGER.error(f"An error occurred: {e}\n", exc_info = True)

    finally:

        LOGGER.info("Exiting...")