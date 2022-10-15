"""Test that TensorFlow can access GPU."""
import tensorflow as tf


def test_gpu() -> None:
    """Test that TensorFlow can access GPU."""
    devices = tf.config.list_physical_devices("GPU")
    assert len(devices) > 0
    print("Found GPU device(s)!")
    for device in devices:
        print("- ", device)
