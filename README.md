# codes
First code "split_by_driver_simple_model_V1" splits data by driver ids in train and valid set <br />
A simple keras based convolution model is included for training and test with validation set.


Instructions for how the code for this project is organized.<br />
The raw data can be downloaded from Kaggle StateFarm competition page. The data was extracted and organized as provided by Kaggle.<br />
Note that we used Google Colaboratory Engine to build our models to take advantage of the free GPU provided.  So to use this code in it’s entirely as is, it must be used in Google Colab environment through Google Drive. <br />
However, one could still run the code in standalone computer, using Python or Ipython interpreter.  In such case, the code chunk that is used to connect to Google Colab should just be ignored or deleted.<br />
The process followed for this code was as follows:<br />
1.	The first file to use is create_train_val_sets_for_colab.py  <br />
a.	This script reads loads the raw data and splits the data based on randomly selected train /validation set. <br />
b.	The data can then be saved in pickle format to retrieve later or upload to google drive efficiently to be used with Colaboratory scripts.<br />
c.	The data is saved with names that easily identify the valid ids set used to generate the train/validation data.  <br />
d.	Note minor image processing is done as part of loading the data.<br />

2.	Use any of the various model specific Ipython scripts to train and test models. There are scripts for:<br />
a.	Logistic Regression <br />
b.	Simple NNet and a Simple CNN (in one script)<br />
c.	Deep CNN network<br />
d.	PCA based models.<br />

3.	All scripts are currently coded to read pickled data set generated by running the first script.  You simple have to matchup the valid ids set # to select the right data set to train with.<br />
4.	All scripts provide options to save results and models using the same valid ids set based naming scheme for easy identification. 
