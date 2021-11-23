# Parking Detector
(for school course ANO II)

Using convolutional neural networks to detect empty/full parking spots.(Python w/ Torch & OpenCV2)



To run this project, you need train_images and test_images folders, filled with the images from school. 

To run venv from windows, you may need to open your cmd/PowerShell as administrator and set execution policy: `Set-ExecutionPolicy RemoteSigned`

How to run locally:
- create venv: `python3 -m venv my_venv` 
- run venv: `source my_venv/bin/activate`(linux) or `.\my_venv\Scripts\Activate.ps1`(win)
- install requiremets: `pip install -r .\requirements.txt`
- get PyTorch: https://pytorch.org/get-started/locally/
- optional: get CUDA toolkit https://developer.nvidia.com/cuda-downloads (runs faster than on CPU)
- run main: `python app/main.py`
- ???
- PROFIT