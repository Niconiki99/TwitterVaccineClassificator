# Twitter Vaccine Classificator

## Overview

Twitter Vaccine Classificator is a project aimed at classifying tweets related to vaccines. 
It utilizes natural language processing (NLP) techniques and machine learning to categorize tweets into relevant categories.

## Features

- **Tweet Classification:** The system classifies tweets into different categories related to vaccines, such as positive, negative, or neutral sentiments.
- **Pretrained Model:** The project uses a particular version of BERT found in [https://github.com/marcopoli/AlBERTo-it] , it's a pretrained BERT already traind on italian tweets, so it's efficient on the classification.
- **Network Analysis:** It is also possible to perform network analysis alongside NLP analysis. This involves constructing a graph and dividing it into communities. Additionally, users can be placed in a user space based on their positions using the ForcaAtlas2 algorithm.
 ![Network Visualization](https://github.com/Niconiki99/TwitterVaccineClassificator/raw/main/images/net_ld.png)
- **Multitabular Analysis:** In order to significantly increase the accuracy of our classification, this project is based on a machine learning method that enables the combination of NLP embeddings generated by BERT with other features, whether they are categorical or numerical. These features are based on network-related data, such as the user's community and their position in the user space. Utilizing the powerful [https://github.com/georgian-io/Multimodal-Toolkit] library, we have implemented various methods to concatenate the NLP embeddings with the non-NLP features.
## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Niconiki99/TwitterVaccineClassificator.git
    cd TwitterVaccineClassificator
    ```

2. Install dependencies:
Verify the dependencies on requirements.txt and install what is missing.

## Usage


## Configuration
