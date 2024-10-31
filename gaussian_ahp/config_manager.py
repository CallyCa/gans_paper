import yaml

class ConfigManager:
    """Manage configuration settings from a YAML file."""

    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_config(self, section):
        return self.config.get(section, {})