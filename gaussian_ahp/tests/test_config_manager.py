import pytest
from src.config_manager import ConfigManager

@pytest.fixture
def config_manager():
    return ConfigManager('settings/ahp_settings.yaml')

def test_get_config(config_manager):
    consistency_index = config_manager.get_config('consistency_index')
    assert '1' in consistency_index
    assert consistency_index['1'] == 0

def test_missing_section(config_manager):
    non_existent = config_manager.get_config('non_existent')
    assert non_existent == {}

pytest.main()
