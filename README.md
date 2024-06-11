# MQC
This is the code & dataset respository of the WWW paper 'Asking Multimodal Clarifying Questions in Mixed-Initiative Conversational Search'.

# Dataset
All the dataset files are stored in the data/ folder. 

Answer_data.csv is the full version of the Melon dataset. All images can be accessed via the link.

question_bank.csv stores the all the clarifying questions. 

facet_data/ stores all the qrels.

cwdocs/ is the folder of the documents for each facet, used for retrieval evaluation. 

**All the images can be assess by the format of https://xmrec.github.io/mturk_images/all_images/{img_id}.**

# Code
We release the generative document retrieval code. All the training instances are preprocessed by BM25 first-phase-retrieval and a Bert-QPP clarifying question selection module. 

# Multimodal taxonomy classes
We show the definition and real cases of these multimodal taxonomy via the MQC taxonomy file in the data/ folder.


