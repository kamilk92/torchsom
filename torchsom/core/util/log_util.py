import logging.config, yaml


def setup_logging(logging_conf_path: str):
    with open(logging_conf_path) as logging_config_file:
        config = yaml.safe_load(logging_config_file.read())
    logging.config.dictConfig(config)
