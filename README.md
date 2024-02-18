# Twitter Vaccine Classificator

## Overview

Twitter Vaccine Classification is a project aimed at classifying tweets related to vaccines. 
It utilizes natural language processing (NLP) techniques and machine learning to categorize tweets into relevant categories.

## Features

- **Tweet Classification:** The system classifies tweets into different categories related to vaccines, such as positive, negative, or neutral sentiments.
- **Pretrained Model:** The project uses a particular version of BERT found in [https://github.com/marcopoli/AlBERTo-it], it's a pre-trained BERT already trained on Italian tweets, so it's efficient on the classification.
- **Network Analysis:** It is also possible to perform network analysis alongside NLP analysis. This involves constructing a graph and dividing it into communities. Additionally, users can be placed in a user-space based on their positions using the ForcaAtlas2 algorithm.
- **Multitabular Analysis:** To significantly increase the accuracy of our classification, this project is based on a machine learning method that enables the combination of NLP embeddings generated by BERT with other features, whether they are categorical or numerical. These features are based on network-related data, such as the user's community and position in the user space. Utilizing the powerful [https://github.com/georgian-io/Multimodal-Toolkit] library, we have implemented various methods to concatenate the NLP embeddings with the non-NLP features.

  <div style="display: flex; justify-content: center;">
    <img src="https://github.com/Niconiki99/TwitterVaccineClassificator/raw/main/images/net_ld.png" alt="Leiden Network Visualization" width="400"/>
    <img src="https://github.com/Niconiki99/TwitterVaccineClassificator/raw/main/images/net_lv.png" alt="Louvain Network Visualization" width="400"/>
    </div>
## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Niconiki99/TwitterVaccineClassificator.git
    cd TwitterVaccineClassificator
    ```

2. Install dependencies:
Verify the dependencies on requirements.txt and install what is missing.

## Usage
The project is made up of several modular components, each designed to perform specific analyses independently. To conduct a comprehensive analysis, it is recommended to start with the build_graphs module. This step begins with raw tweets and selects only those that fall within a specific deadline. The module then constructs the graph, laying the foundation for subsequent analyses.

After graph construction, the build_communities module is used to examine the community structures within the graph. This step is critical in understanding the network's intrinsic groupings and relationships. Different algorithms can be used to build the communities.

The network.py module is responsible for computing the positions of user IDs within the graph. It also produces a visual representation of the network, assigning distinct colours to each community. This visualization helps interpret the network's structural dynamics and identify the relationships between different user communities. Together with the information of the communities, the positions are the key features to couple with the embeddings performed by BERT to classify the tweets.

The next stage involves preprocessing tasks such as dataset merging, splitting, and other preparatory steps. The Preprocessing module manages this crucial phase, ensuring that the data is appropriately formatted and ready for subsequent analyses. This module couples all the features produced by the previous modules with the raw tweets.

Finally, the project includes the MultiBERT_train module, which focuses on training models based on the preprocessed data. This step allows for the development and refinement of models that can provide insights and predictions based on the characteristics of the network and user communities.
# BuildGraph Parameters
- **DATAPATH:** Path to the folder storing the dataset.
- **deadline:** A specified deadline for parsing, set to "2021-06-01".

# BuildCom Parameters
- **NETPATH:** Path to the folder storing network-related data, also where the build_graph module saves the graphs.

# Preprocess Parameters
- **TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR, NETWORK_DATA:** Paths to different data directories.
- **path_df:** Path to the CSV file containing the main DataFrame (`df_full`).
- **name_df, dtype_df:** Column names and data types for the main data frame.
- **path_com:** Path to the CSV file containing community-related data.
- **dtype_com, names_com:** Data types and column names for community-related data.
- **path_pos:** Path to the JSON file storing position information.
- **names_pos:** Names for x and y position columns.
- **labels:** List of labels required for machine learning.
- **random_state:** Seed for random state.

# Network Parameters
- **MAKE:** Boolean indicating whether to recalculate user positions using fa2.
- **com2col:** List of colours used for drawing communities.
- **path_to_save, path_to_read:** Paths for saving and reading position information.

# Machine Learning Parameters
- **bert:** Pre-trained BERT model for text processing.
- **train_path, val_path, test_path:** Paths to training, validation, and test datasets.
- **names_dataset, dtype_dataset:** Parameters for loading the training, validation and test datasets.
- **text_cols, categorical_cols, numerical_cols:** Lists specifying text, categorical, and numerical columns.
- **categorical_encode_type, numerical_transformer_method:** Encoding methods for categorical variables and transformation method for numerical variables.
- **label_col, label_list:** Column name for labels and list of possible label values.
- **dataset_params:** Tuple containing parameters for loading the dataset.
- **combine_feat_method, cat_feat_dim, numerical_feat_dim:** Configuration parameters for combining features in tabular data.
- **tab_conf_params:** Tuple containing tabular configuration parameters.
- **use_cpu, overwrite_output_dir, do_train, do_eval, per_device_train_batch_size, num_train_epochs, logging_steps, eval_steps, weight_decay, auto_find_batch_size, dataloader_drop_last:** Training arguments and parameters, required by MULTIMODALToolkits structure.

## Requirements: 
The code is built on python 3.9 and is based on some fundamental packages:
- **numpy**
- **pandas**
- **networkx**
- **scipy**
- **pathlib**
- **collections**
- **igraph**
- **sknetwork**
- **sklearn**
- **re**
- **os**
- **torch**
- **transformers**
- **multimodal toolkit (https://github.com/georgian-io/Multimodal-Toolkit.git)**
- **json**
- **matplotlib**
- **time**
- **forceatlas2 (https://github.com/AminAlam/forceatlas2.git)**
