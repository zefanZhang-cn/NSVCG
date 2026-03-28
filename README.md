## NSVCG
Source code of our paper entitled "Non-Salient Visual Content Grounding for Multimodal Relation Extraction".


## Required Environment
To run the codes, you need to install the requirements for [RE](requirements.txt).

    pip install -r requirements.txt

## Data Preparation
* MNRE
  
  You need to download three kinds of data to run the code.  
  > 1.The raw images of [MNRE](https://github.com/thecharm/MNRE), many thanks.  
  > 2.The visual objects from the raw images from [HVPNeT](https://github.com/zjunlp/HVPNeT), many thanks.  
  > 3.The generated image features from [InstructBLIP]([https://github.com/thecharm/TMR](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)), many thanks.
  
  Then you should put folders ``img_org``,  ``img_vg``  under the "./data" path.



## Acknowledge
Sincerely thanks to [***InstructBLIP***](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) and [***TMR***](https://github.com/thecharm/TMR) for their contributions to this study. Undoubtedly, our success is inseparable from the efforts of any researcher who focuses on multimodal relation extraction tasks. Finally, we sincerely wish every researcher has wonderful scientific research!
