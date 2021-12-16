# A Graph-Based Neural Model for End-to-End Frame Semantic Parsing
This repository is the pytorch implementation of our paper:

**A Graph-Based Neural Model for End-to-End Frame Semantic Parsing**<br>
Zhichao Lin, Yueheng Sun, [Meishan Zhang](http://zhangmeishan.github.io//) <br>
The Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021<br>

**!!!Note: we have upgrade our codebase to allennlp 2.0+ which is different from our origin implementation based on alennnlp 1.0. We find that our model can achieve further performance about 49.20 ROLE-F1 in Framenet1.5 and 49.05 ROLE-F1 in Framenet 1.7.**
## Installation 

Clone the current repository:
```bash
git clone https://github.com/Ch4osMy7h/FramenetParser.git
cd FramenetParser
```

### Allennlp and Other Dependencies


```bash
# Create python environment (optional)
conda create -n framenet python=3.7
source activate framenet

# Install python dependencies
pip install -r requirements.txt
```

### Data
Request and download Framenet Dataset 1.5 and Framenet Dataset 1.7 from [Framenet Official Website]("https://framenet.icsi.berkeley.edu/fndrupal/"). Place them under `data` directory, which contatins two subdirectories called `fndata-1.5` and `fndata-1.7`. Specially, the data should be splited into there subdirectories following [FN1.5](https://github.com/swabhs/scaffolding/blob/master/allennlp/data/dataset_readers/framenet/full_text_reader.py) and [FN1.7](https://github.com/swabhs/open-sesame/blob/625da3d451/sesame/globalconfig.py). Structures are showed below:

```bash
./data
    fndata-1.5/
        train/ # containing the training xml files
        dev/
        test/
        ...
    fndata-1.7/
        train/
        dev/
        test/
        ...
```

To get the datasets our code based on, run the following commands:
```bash
python preprocess.py
```

## Training & Evaluation
We use train_parser.sh script to train and evaluate all of our joint models. You can modify any configuration you want in the script.
```bash
bash train_parser.sh
```

### Evaluation Only
All models can be evaluated using by the following commands:
```bash
allennlp evaluate your_evaluation_model \
    your_evaluation_data_file \
    --include-package framenet_parser \
    --output-file your_evaluation_output.json \
```

## Inference on unlabed data
For predicting targets, frames and frame element on your own unlabed data (follow the file format as shown in `experiments/inference/sample.json`), you can use the following commands:
```bash
allennlp predict \
  --output-file your_wanted_output_file \
  --include-package framenet_parser \
  --predictor framenet_parser \
  --cuda-device 0 \
  your_trained_model \
  your_own_data.json
```

## TODOs:
- [ ] An easy-to-use pipeline models for specific usages (will be published recently)
- [x] An easy-to-use full joint model for extracting frame-semantic structures
