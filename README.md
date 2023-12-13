# The online repository for the under review IJCAI2024 paper "GTE: A Framework for Learning Code AST Representation Efficiently and Effectively"

## Data Preprocess
Download the divided data from https://doi.org/10.5281/zenodo.10369493

Run `process_data_PC.py` to get preprocessed code graph data for program classification tasks, for example:
```Shell
python preprocess/process_data_PC.py --raw_dir <YOUR_DATA_DIR>/Project_CodeNet_Java250_RATIO6-2-2 --language java --output_dir <YOUR_DATA_DIR>
```
Run `process_data_PT.py` to get preprocessed code graph data for probing tasks, for example:
```Shell
python preprocess/process_data_PT.py --raw_dir <YOUR_DATA_DIR>/Java_MINsample200_MAXline200_Accepted_RATIO8-1-1 --language java --output_dir <YOUR_DATA_DIR>
```

After that, a new directory which name ends with `_DGL` contains the preprocessed data will be generated in <YOUR_DATA_DIR>.

## Training
Run `GTE.py` in the `Experiment` directory to training the GTE-based models, for example:
```Shell
python Experiment/ProgramClassificationTransformer/GTE.py
```
The model hyperparameters can be found in `configuration.yaml`.
You may want to train different GTE-based models, please change the `from models.GTE_model_Transformer import GTEProgramClassification` to other models in `models` directory.
