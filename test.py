import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="test.yaml", version_base="1.2")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.testing_pipeline import test, baseline_test

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    if ".baseline." in config.model._target_: # if baseline model, no need to run AI pipeline
        return baseline_test(config)
    else:
        return test(config)


if __name__ == "__main__":
    main()
