# CNN for Music Audio Tagging

A PyTorch implementation of the musicnn model by Jordi Pons [1], a CNN-based audio feature extractor and tagger.
This implementation is still a WIP and does not strictly follow the musicnn architecture (e.g. so far only one type of convolutional filter is used in the first layer of the front end, instead of the five different types in the original model).

## Installation
```
git clone https://github.com/ilaria-manco/dl4am
```
Create a virtual environment and activate it
```
bash
python3 -m venv env
source venv/bin/activate
```
Install the required dependencies 
```
bash pip install -r requirements.txt 
```
## Training the model
If you want to retrain the model on the MTT dataset, you'll have to download this first from [here](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset). After doing this, change ```config_file.py``` to point to the correct paths where the data is stored and then preprocess the audio files by running
```
python run_preprocessing.py --mtt         
```
Then you can use the following script for training, after changing the parameters in ```config_file.py```, if necessary.
```
python run_training.py         
```

## Using the pre-trained model
This repo also contains 3 pre-trained model ready to use. For example, you can extract the outputs tags by running
```
python extract_features.py --input_audio --output_path --model_number    
```
For model_number, 2 is the one found to perform better in the preliminary evaluation and is therefore recommended.

## Evaluating the model
The evaluation script computes two metrics, mean ROC AUC and mean PR AUC and produces a plot of the two metrics over the TFR vs FPR. The evaluation is done on 5328 test data samples from the MTT. 
```python evaluate.py --model_number    
```

## References
[1] Pons, Jordi, and Xavier Serra. "musicnn: Pre-trained convolutional neural networks for music audio tagging." arXiv preprint arXiv:1909.06654 (2019).
