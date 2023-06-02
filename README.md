
## TEM4CTR (Time-aligned Exposure-enhanced Model for Click-Through Rate Prediction)    


## Description   
Click-Through Rate (CTR) prediction task plays a crucial role in various applications, such as recommender systems and online advertising, and is typically applied for the ranking of items. The objective of the CTR prediction task is to predict whether a user will click on a given candidate item. One of the remarkable milestones in CTR prediction methods is the introduction of user behavior sequence modeling. These works typically capture the users' latent interest from their history behavior sequences to perform CTR prediction. In recent years, researchers try to use implicit feedback sequence data such as exposured but unclicked records to further help the CTR model better extract the complex and various latent interest of users. However, there are some key problems in these works: the temporal misalignment and lack of fine-grained interactions among feedback sequences. Temporal misalignment implies that there are significant differences in the time ranges corresponding to each sequence, thus limiting the performance improvement of the auxiliary feedback representations for user interest extraction and model performance. Fine-grained information interaction between sequences can further improve the utilization of auxiliary feedback. To address these problems, we propose TEM4CTR to ensure the temporal alignment between sequences, while using the auxiliary feedback information for fine-grained item-level information enhancement on click behavior based on the representation projection mechanism. In addition, this projection-based information transfer module can effectively alleviate the negative impact of useless or even harmful components of the auxiliary feedback information on the learning of click behavior. Comprehensive experimental results on public and industrial datasets validate the significant superiority and effectiveness of our proposed TEM4CTR, as well as illustrate the important research implications of temporal alignment in multi-feedback modeling.

## Datasets
Download the datasets via the following url and move the datasets to ./data folder.
- [Alibaba](https://tianchi.aliyun.com/dataset/56)
- [Microvideo-1.7M](https://github.com/Ocxs/THACIL)

## How to run   
First, install dependencies   
```bash  
cd TEM4CTR
pip install -r requirements.txt
```

 Next, run the pre-processing code for corresponding datasets
 ```bash
 python code/preprocess_alibaba_align.py 
 python code/preprocess_microvideo_align.py
 ```

 Then, run the model for corresponding datasets
  ```bash
  python code/train_alibaba_feedback.py -p train
  python code/train_microvideo_feedback.py -p train
  ```