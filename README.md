# age_model_tfapi
This is the code based on [AEC_model](https://github.com/Soyuen/age_estimation_compact_model), and using tf.data API to optimize tensorFlow GPU performance.
## Requirements
* Anaconda
* Python 3.7
* [Packages](https://github.com/Soyuen/age_estimation_compact_model/blob/main/packages.txt)

### Install packages
```
pip install -r packages.txt
```
## Procedure
### Data processcing I
If you don't want to spend much time downloading these datasets and doing data processcing, you can skip these steps and doing **Data processcing II**.
* Download [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar) and [WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar) datasets.(https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
* Extract files to "./data"
* Run the code below for data processing
```
cd ./data
python imdbwiki_filter.py
python tfrecord.py
python imdbwiki_filter.py --db wiki
python tfrecord.py --db wiki
```
### Data processcing II
If you were already doing **Data processcing I**, you can skip these steps.  
*Run the code below for merge imdb.tfrecords
```
cd ./data
python merge_record.py
```
### Training  model
Training model with 90 epochs.The batch size is 128 on Imdb dataset and 50 on Wiki dataset.(the same setting with SSR-Net)
```
cd ./training
python train.py --input ../data/imdb.npz --db imdb
python train.py --input ../data/wiki.npz --db wiki  --batch_size 50
```
Using original code  
<img src="https://github.com/Soyuen/picture/blob/main/gpu1.JPG" width = "400" height = "400" alt="result">  
Using this code  
<img src="https://github.com/Soyuen/picture/blob/main/gpu2.JPG" width = "400" height = "400" alt="result">  


|  Method  | Imdb (training time)  | Wiki (training time)  | volatile GPU-util |
|----------|:---------------------:|:---------------------:|:-----------------:|
| Original |   1 hr 56 mins        |        27 mins        |       41%         |
|This code |       31 mins         |        10 mins        |       83%         | 

This code is training **triple** faster than original code.
