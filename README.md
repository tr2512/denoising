# Image denoising - Computer vision capstone project
To setup the repositories, install the requirements
```sh
pip install -r requirements.txt
```
Download the checkpoint from the link: https://drive.google.com/drive/folders/1GKRGEYPmlPHDb5wvNa_sBMDIY_N0XEs-?usp=sharing  
Put the checkpoint in the same directory as this github.
## Run inference on interface
```sh
gradio app.py
```
After that, access localhost:7860 to use the denoising interface.
Or you can use the demo we deployed at HuggingFace: https://huggingface.co/spaces/tr251202/CV-project

## Train Invertible Denoising Network
Download SIDD medium dataset from :https://www.eecs.yorku.ca/~kamel/sidd/
Extract the dataset and put into folders: train/gt; train/noisy, va;/gt, val/nosiy
Run the command:
```sh
python train.py -imgtraindir <train_noisy> --lbltraindir <train_gt> --imgvaldir <val_noisy> --lblvaldir <val_gt>
```