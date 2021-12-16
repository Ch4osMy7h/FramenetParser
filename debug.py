import logging
import os
import sys

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)


from allennlp.commands import main

experiment_name="debug"
config_file="./training_config/debug.jsonnet"
cuda_device=5

sys.argv = [
    "allennlp",
    "train",
    config_file,
    "--serialization-dir", "./experiments/training/{}".format(experiment_name),
    "--include-package", 'framenet_parser',
    "-f"
]

main()

