from huggingface_hub import ModelCard

def save_model_card(content: str, output_path: str = "merge/README.md") -> None:
    """Saves the model card content to a file.

    Args:
        content: The model card content as a string.
        output_path: The path to save the model card (default: 'merge/README.md').
    """
    try:
        card = ModelCard(content)
        card.save(output_path)
    except Exception as e:
        print(f"Error saving model card: {e}")
