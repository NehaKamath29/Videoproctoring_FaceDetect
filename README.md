this model performs multiple face detection, pos detection and also differentiates photo frame and real faces.
**Steps:**
1.Run gather examples. All the images should be generated in the live and non_live folders accordingly.
├───liveness_dataset
│   ├───live
│   └───non_live
the above is the dataset
├───liveness_output_img
│   ├───live
│   └───non_live
the above is the folder where the gathered images will be stored
(note: dataset is not present in the repo, add the folders for dataset accordingly)
2.Run train.py. Because you are running for first time, le.pickle file and liveness_model.h5 will be generated, for all other times you run train.py, these two files will only get updated.So you dont have to create these two files in the start. They will get generated.
3.Run head_posing.py.
