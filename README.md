# CNN for Music Audio Tagging (WIP)

A PyTorch implementation of the musicnn model by Jordi Pons [1], a CNN-based audio feature extractor and tagger.
This implementation is still a WIP and does not strictly follow the musicnn architecture (e.g. so far only one type of convolutional filter is used in the first layer of the front end, instead of the five different types in the original model).

## Installation
```
git clone https://github.com/ilaria-manco/music-audio-tagging-pytorch
```
Create a virtual environment and activate it
```
python3 -m venv env
source venv/bin/activate
```
Install the required dependencies 
```
pip install -r requirements.txt 
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

## Evaluating the model
The evaluation script computes two metrics, mean ROC AUC and mean PR AUC and produces a plot of the two metrics over the TFR vs FPR. The evaluation is done on 5328 test data samples from the MTT. 
```
python evaluate.py --model_number    
```

## Using the pre-trained model
This repo also contains 3 pre-trained models ready to use. 

### Extract the output features
You can extract the output tags by running
```
python extract_features.py --input_audio --output_path --model_number    
```
For model_number, 2 is the one found to perform better in the preliminary evaluation and is therefore recommended.
### Get the top N tags
An example of sample recognition can be found in this [Jupyter notebook](https://github.com/ilaria-manco/music-audio-tagging-pytorch/blob/master/src/Sample%20Recognition.ipynb)
```
python extract_features.py --input_audio --num_samples    
```

## References
[1] Pons, Jordi, and Xavier Serra. "musicnn: Pre-trained convolutional neural networks for music audio tagging." arXiv preprint arXiv:1909.06654 (2019).
