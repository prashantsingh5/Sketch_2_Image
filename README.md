# Sketch_2_image

## Installation

1. **clone the repository**:
   ```bash
   git clone https://github.com/prashantsingh5/Sketch_2_image.git
   ```

2. **Move to pytorch-CycleGAN-and-pix2pix**:
   ```bash
   cd pytorch-CycleGAN-and-pix2pix
   ```

3. **Install Dependency**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset**:
   In base folder add a new datset folder with the given structure

  ```plaintext
dataset/                     
│
├── subfolder/                      
    ├── img1.jpg         
    ├── img2.jpg          

```

2. In the base folder **Move to script**:
   ```bash
   cd script
   ```

3. **Update the path inside process_dataset.py**:
   - base_folder = r'update with the path of base_code'
   - pytorch_folder = r'update with the path of base_code\pytorch-CycleGAN-and-pix2pix'
  
4. **Run process_dataset.py**:
   ```bash
   python process_dataset.py
   ```

This code is creating 2 new folders:
- **dataset_Sketch**: - This folder has sketch images.
- **dataset_for_pix2pix**: - This folder has the necessary structure for pix2pix.

5. **Run sketch2image_model_train.ipynb**:
   - Run all the cells in sketch2image_model_train.ipynb file that will generate a best model that is further used for sketch to image conversion
   - After sucessfully running all the cells this will generate the best model.
  
6. **From the base directory move to pytorch-CycleGAN-and-pix2pix**:
   ```bash
   cd pytorch-CycleGAN-and-pix2pix
   ```
   
7. **Run test_sketch_to_image.py file**:
   ```bash
   python test_sketch_to_image.py
   ```

8. **The above file will return the output real image in new folder named "output_images". The output folder is inside  pytorch-CycleGAN-and-pix2pix.**
   
