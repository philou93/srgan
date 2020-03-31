from tensorflow.python.lib.io import file_io


def history_memory(func):
    """
      Décorateur qui garde en mémoire l'historique des loss du discriminateur et générateur.
      On en a besoin parce que si on utilise un décorateur de degré 1, la mémoire va pas être partagé entre fonction.
      :param func: callable
      :return:
      """
    generator_history = []
    discriminator_history = []

    def outer_wrapper(f):
        def wrapper(*args, **kwargs):
            nonlocal generator_history, discriminator_history
            return func(f(generator_history, discriminator_history, *args))
        return wrapper
    return outer_wrapper


@history_memory
def act_on_history(func):
    """
    Appele la fonction qui veut agir sur l'historique avec les arguments de cette fonction.
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        func(*args)
    return wrapper


@act_on_history
def add_loss_to_history(generator_history, discriminator_history, generator_loss, discriminator_loss):
    """
    Ajoute les loss à l'historique.
    :param generator_history:
    :param discriminator_history:
    :param generator_loss:
    :param discriminator_loss:
    :return:
    """
    generator_history.append(generator_loss)
    discriminator_history.append(discriminator_loss)


@act_on_history
def flush_loss_history(generator_history, discriminator_history, path):
    """
    Enregistre les valeurs dans les listes des historiques et les resets.
    :param generator_history:
    :param discriminator_history:
    :param path:
    :return:
    """
    if not file_io.file_exists(path):
        file_io.write_string_to_file(path, "")  # On crée le fichier
    with file_io.FileIO(path, mode='w') as input_f:
        for i in range(len(generator_history)):
            input_f.write(f"{generator_history[i]},{discriminator_history[i]}\n")
    print(f"{len(generator_history)} of history element will be flush.")
    del generator_history[:]
    del discriminator_history[:]
    print(f"History flush.")
