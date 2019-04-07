# Machine Learning Engineer Nanodegree

# Capstone project

## Software and Libraries

This project has been implemented using Atom as Text Editor to create the `report.md` file as well as converting to PDF `report.pdf`. And `Jupyter Notebook` is used to host the code implementation for the project, along with the following Libraries used during the process to help develop the model:

1. NumPy
2. Pandas
3. Seaborn
4. Matplotlib
5. OpenCV
6. Keras APIs
7. tqdm (for better displaying backend)
8. SciPy loadmat (for loading `.mat` format)
9. Scikit-learn (for Metrics)
10. Python 3 (as main language)

The project has been setup along with separate environment named `capstone-env` via `Virtual environment` to ensure best practices when develop program using Python.

Flower data images can be <a target="_blank" href="https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip">downloaded from here</a>. This is from PyTorch Schlarship challenge, which has already been grouped into different classes from **1 to 102** categories. After downloading this dataset, unzip and put it into the same folder `flower_data`, which has already been created to host another file named `cat_to_name.json` in this repository.

After all, observe the implemented solution in the notebook `flower-keras-khang` from `Notebooks` folder. If a final model is desired to be built, then execute the entire notebook end-to-end to have the `final_model.h5` file created in the `models` folder. However, this file only contains the final weights for the final model for memory efficiency.
