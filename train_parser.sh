SEEDS=(233)

# Framenet 1.5
train_data_path="data/preprocessed-fn1.7/train.json"
dev_data_path="data/preprocessed-fn1.7/dev.json"
test_data_path="data/preprocessed-fn1.7/test.json"
ontology_path="data/fndata-1.7/"

config_file="./training_config/framenet_parser.jsonnet"
cuda_device="0"

# proposed approach
for seed in ${SEEDS[@]}; do
    train_data_path=$train_data_path \
    dev_data_path=$dev_data_path \
    test_data_path=$test_data_path \
    ontology_path=$ontology_path \
    cuda_device=$cuda_device \
    seed=$SEEDS \
        allennlp train $config_file \
        $hyper_file \
        --serialization-dir ./experiments/training/framenet_parser_v1.7_${seed} \
        --include-package framenet_parser \
        -f
    done
done