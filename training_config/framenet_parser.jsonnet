local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);


local bert_model = "bert-base-cased";
local max_length = 128;
local feature_size = 20;
local max_span_width = 15;

local bert_dim = 768;  # uniquely determined by bert_model
local lstm_dim = 200;

local seed = std.parseInt(std.extVar('seed'));
local cache = std.extVar('cache_directory');
local ontology_path = std.extVar('ontology_path');

local span_embedding_dim = 2 * lstm_dim + bert_dim + feature_size;
local span_pair_embedding_dim = 3 * span_embedding_dim;

local task = "joint";
local display_metrics = {
    "node": ["node_type_f1", "node_attr_f1"],
    "edge": ["p2p_edges_f1", "p2r_edges_f1"],
    "joint": ["target_f1", "frame_f1", "role_f1"]
  };

{
  "dataset_reader": {
    "type": "framenet",
    "ontology_path": ontology_path, 
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": bert_model,
        "max_length": max_length
      },
    },
    "max_span_width": max_span_width
  },
  "train_data_path": std.extVar("train_data_path"),
  "validation_data_path": std.extVar("dev_data_path"),
  "test_data_path": std.extVar("test_data_path"),
  "model": {
    "type": "framenet_parser",
    "display_metrics": display_metrics[task],
    "loss_weights": {
      "node": 1.0,
      "edge": 1.0,
    },
    "ontology_path": ontology_path,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": bert_model,
            "max_length": max_length
        }
      }
    },
    "context_layer": {
      "type": "alternating_lstm",
      "input_size": bert_dim,
      "hidden_size": lstm_dim,
      "num_layers": 6,
      "recurrent_dropout_probability": 0.1,
      "use_input_projection_bias": true
    },
    "modules": {
      "node": {
        "node_feedforward": {
          "input_dim": span_embedding_dim,
          "num_layers": 2,
          "hidden_dims": 150,
          "activations": "relu",
          "dropout": 0.2,
        },
      },
      "edge": {
        "predicate_ratio": 0.6,
        "role_ratio": 0.8,
        "predicate_mention_feedforward": {
          "input_dim": span_embedding_dim,
          "num_layers": 1,
          "hidden_dims": 150,
          "activations": "relu",
          "dropout": 0.2,
        },
        "role_mention_feedforward": {
          "input_dim": span_embedding_dim,
          "num_layers": 1,
          "hidden_dims": 150,
          "activations": "relu",
          "dropout": 0.2,
        },
         "p2p_edges_feedforward": {
          "input_dim": 3 * 150,
          "num_layers": 1,
          "hidden_dims": 150,
          "activations": "relu",
          "dropout": 0.2,
        },
        "p2r_edges_feedforward": {
          "input_dim": 3 * 150,
          "num_layers": 1,
          "hidden_dims": 150,
          "activations": "relu",
          "dropout": 0.2,
        }
      },
    },
    "initializer": {
        "regexes": [
            [".*linear_layers.*weight", {"type": "xavier_normal"}],
            [".*scorer._module.weight", {"type": "xavier_normal"}],
            ["_distance_embedding.weight", {"type": "xavier_normal"}],
            ["_span_width_embedding.weight", {"type": "xavier_normal"}],
            ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
            ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
        ]
    },
    "lexical_dropout": 0.5,
    "lstm_dropout": 0.4,
    "feature_size": feature_size,
    "max_span_width": max_span_width
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["text"],
      "batch_size": 8,
    }
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed,
  "trainer": {
    "num_epochs": 250,
    "grad_norm": 5.0,
    "patience" : 20,
    "cuda_device": std.parseInt(std.extVar("cuda_device")),
    "validation_metric": "+role_f1",
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 4
    },
    "checkpointer": {
      "keep_most_recent_by_count": 1,
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-3,
      "weight_decay": 0.01,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    }
  },
  "evaluate_on_test": true
}