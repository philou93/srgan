import tensorflow as tf


def setup_device_use():
    """
    Dépendament si on a accès à o, 1 ou 2 gpus, on affecte un gpu à un des modèles.
    :return: un dictionnaire qui map les modèles au device sur lequel il va être exécuté.
    """
    def _get_device_name(path):
        return path[-5:]

    model_to_gpus_map = {
        "descriminator": "/cpu:0",
        "generator": "/cpu:0"
    }
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) >= 2:
        print(f"Descriminator will be map to to device {_get_device_name(gpus[0].name)}.")
        model_to_gpus_map["descriminator"] = _get_device_name(gpus[0].name)
        print(f"Generator will be map to to device {_get_device_name(gpus[1].name)}.")
        model_to_gpus_map["generator"] = _get_device_name(gpus[1].name)
    elif len(gpus) == 1:
        print(f"Descriminator will be map to to device {_get_device_name(gpus[0].name)}.")
        model_to_gpus_map["descriminator"] = _get_device_name(gpus[0].name)
        print(f"Generator will be map to to device {_get_device_name(gpus[0].name)}.")
        model_to_gpus_map["generator"] = _get_device_name(gpus[0].name)

    return model_to_gpus_map


def set_to_memory_growth():
    """
    On évite de tout mettre d'un coup dans le gpu. (Pas certain de l'efficacité)
    :return:
    """
    print("Setting GPUs memory growth...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth set for GPU: {gpu.name}")
