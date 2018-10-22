# TextSum
Focal Point 1: Text Summarization

## Guide to using the project
### Installation
* You can try using conda create --name <env> --file req.txt
* If that does not work (usually due to os differences, etc), create a conda environment and install the following packages:
    * nltk
        * You should also need to run nltk.download() (and then download the punkt encoder/dataset) in a terminal window or by
        opening the Initial Data Exploration Notebook
    * jupyter
    * scikit-learn

### Use
* Run clean_data.py to generate the cleaned dataset
* Run word2vec.py (with --model_name=\<insert_model_name_here\>)
* Run textsum.py with --model_name=\<insert_model_name_here\>)