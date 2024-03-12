
# Project: face detection for video protoring

## Steps:

1. **Run gather examples:** 
   All the images should be generated in the `live` and `non_live` folders accordingly.
   ├───liveness_dataset
   │ ├───live
   │ └───non_live
   The above is the dataset tree.
   ├───liveness_output_img
   │ ├───live
   │ └───non_live
      The above is the folder where the gathered images will be stored.

   *Note: Dataset is not present in the repo; add the folders for the dataset accordingly.*

2. **Run train.py:** 
   Because you are running for the first time, `le.pickle` file and `liveness_model.h5` will be generated. For all other times you run `train.py`, these two files will only get updated. So you don't have to create these two files in the start. They will get generated.

3. **Run head_posing.py.**


## Notes:

- Ensure all necessary folders are present before running the scripts.
- Make sure to check the generated files after running the scripts.





