from sacred import Experiment

ex = Experiment("Transformer")

@ex.config
def config():
    seed = 42
    batch = 256
    file_name = "Data"
    run_enviroment = "/home/yangcw/video/Transform"
    label_length = 17
    context_length = 21
    epoch = 100
