import yaml
from typing import List, Dict, Any

def load_yaml_config(yaml_config: str) -> Dict[str, Any]:
    """Loads a YAML configuration file and performs basic validation.

    Args:
        yaml_config: The YAML configuration string.

    Returns:
        A dictionary containing the parsed YAML data.  Raises YAML error if the config fails to parse.
    """
    try:
        config = yaml.safe_load(yaml_config)
        if not isinstance(config, dict):
            raise ValueError("YAML config must be a dictionary.")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

def extract_models(config: Dict[str, Any]) -> List[str]:
    """Extracts model names from a YAML configuration dictionary.

    Args:
        config: The parsed YAML configuration dictionary.

    Returns:
        A list of model names.  Raises ValueError if no models are found in the specified format.
    """
    if "models" in config:
        return [model["model"] for model in config["models"] if "parameters" in model]
    elif "slices" in config:
        return [source["model"] for slice_ in config["slices"] for source in slice_["sources"]]
    else:
        raise ValueError("No 'models' or 'slices' section found in the YAML configuration.")
