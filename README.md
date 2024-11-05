# MQC
This is the code & dataset respository of the WWW paper 'Asking Multimodal Clarifying Questions in Mixed-Initiative Conversational Search'.

# Dataset
All the dataset files are stored in the data/ folder. 

Answer_data.csv is the full version of the Melon dataset. All images can be accessed via the link.

question_bank.csv stores the all the clarifying questions. 

facet_data/ stores all the qrels.

cwdocs/ is the folder of the documents for each facet, used for retrieval evaluation. 

qrels/ is the ground-truth qrel we used for training/validation/inference.

**All the images can be assess by the format of https://xmrec.github.io/mturk_images/all_images/{img_id}.**

# Baselines
The Bert based retrieval code is stored in cedr/. To train, run train.sh. To rerank, run test.sh. 

To add multimodal information, change train.py to train_multimodal.py.

The BM25-based method is stored in first_phase_retrieval. 

bm25.py is to retrieve documents based on topics, questions, and answers.

bm25_ques.py is to retrieve questions based on topics.

gdeval.pl is the evaluation script. To run it, use 
```
perl gdeval.pl [ground-truth.qrel] [result.qrel].
```

# Code

The code for generative retrieval is stored in VL-T5/. For environments, please refer to the original VL-T5 repository.

The link for image features can be downloaded in 

# Multimodal taxonomy classes
We show the definition and real cases of these multimodal taxonomy via the MQC taxonomy file in the data/ folder.

# Citations
If you find this useful, please cite
```
@inproceedings{10.1145/3589334.3645483,
author = {Yuan, Yifei and Siro, Clemencia and Aliannejadi, Mohammad and Rijke, Maarten de and Lam, Wai},
title = {Asking Multimodal Clarifying Questions in Mixed-Initiative Conversational Search},
year = {2024},
url = {https://doi.org/10.1145/3589334.3645483},
doi = {10.1145/3589334.3645483},
booktitle = {Proceedings of the ACM Web Conference 2024},
pages = {1474–1485},
}
```
