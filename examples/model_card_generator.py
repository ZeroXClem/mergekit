from jinja2 import Template
from typing import List

def generate_model_card_content(
        model_name: str, models: List[str], yaml_config: str, username: str
) -> str:
    """Generates the content for a model card using a Jinja template.

    Args:
        model_name: The name of the model.
        models: A list of model names used in the merge.
        yaml_config: The YAML configuration string.
        username: The Hugging Face username.

    Returns:
        The generated model card content as a string.
    """
    template_text = """
    ---
    license: apache-2.0
    tags:
    - merge
    - mergekit
    - lazymergekit
    {%- for model in models %}
    - {{ model }}
    {%- endfor %}
    ---

    # {{ model_name }}

    {{ model_name }} is a merge of the following models using [mergekit](https://github.com/cg123/mergekit):

    {%- for model in models %}
    * [{{ model }}](https://huggingface.co/{{ model }})
    {%- endfor %}

    ## 🧩 Configuration

    ```yaml
    {{- yaml_config -}}
    ```
    """
    template = Template(template_text.strip())
    return template.render(
        model_name=model_name, models=models, yaml_config=yaml_config, username=username
    )
