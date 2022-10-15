"""Clean up CNN environment."""
import gc

from tensorflow import keras


def bye() -> None:
    """Clean up (useful when training multiple models in a loop) and say goodbye."""
    keras.backend.clear_session()
    gc.collect()
    print("Goodbye for now.")
