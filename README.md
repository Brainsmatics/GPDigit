# GPDigit

GPDigit: A Generalized Plaque Digitization Framework for Mesoscopic High-Resolution Images

GPDigit is a comprehensive framework for Aβ plaque digitization, 
designed for mesoscopic high-resolution images and supporting both 2D and 3D data. 
It covers the complete plaque digitization pipeline, including image preprocessing, 
label processing, plaque detection model construction and training, 
plaque foreground signal segmentation, and postprocessing of detection and segmentation results.


## Dependencies

Python >= 3.6.0   
TensorFlow >= 1.9.0   
Torch >= 1.6.0   
Torchvision >= 0.7.0   
To use the label_revise module, please ensure that the public software 3D Slicer 
is installed along with the corresponding Jupyter Notebook extension.

Detailed dependencies for each module can be found in the 'requirements.txt' files 
located in the respective folders.

You can also install the requirements with:
```bash
pip install -r ./requirements.txt
```



## Framework Structure
```text
GPDigit/
├── preprocessing/                  # Data chunking, data augmentation, label conversion and processing
├── label_revise/                   # Annotation tool construction
│                                   # Run labelrevise-note.ipynb to generate the interactive interface
├── 2D_detection/ 
│   │   ├── mrcnn/                  # Network architecture implementation
│   │   ├── samples/                # Ablation studies
│   │   │   └── Palques/            # Model training
│   │   ├── predict_eval/           # Prediction and evaluation
│   │   └── output_process/         # Postprocessing of detection results
│   └── 2D_plaque_detection.h5      # 2D model
├── 3D_detection/ 
│   ├── darknet                     # Network architecture implementation
│   ├── cfg/                        # Network configuration 
│   │   └── yolov3_drop             
│   ├── train                       # Model training
│   ├── predict_evaluation          # Prediction and evaluation
│   ├── 3D_plaque_detection.pth     # 3D model
│   └── ...  
└── segmentation/                   # Plaque signal segmentation: single & blocks.                             
```


## Models

The 2D and 3D plaque detection models are stored in their respective directories.
Parameter settings refer to the current settings of all source code.


## Data

The data for this study are available from the corresponding author upon reasonable request.  





