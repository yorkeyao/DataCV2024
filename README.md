# DataCV2024: The 2nd CVPR DataCV Challenge

Welcome to DataCV2024!

This is the development kit repository for [the 2nd CVPR DataCV Challenge](https://sites.google.com/view/vdu-cvpr24/competition/). Here you can find details on how to download datasets, run baseline models, and evaluate the performance of your model. Evaluation can be performed remotely on the CodeLab evaluation server. Please see the main website for competition details, rules, and dates.

Have fun!

## Overview (NEW)
The 2nd DataCV Challenge is held in conjunction with the CVPR 2024 Visual Dataset Understanding workshop. It is the first of its kind in the community, where we focus on searching training sets for various targets. The competition is held via CodaLab and includes two phases. 

## Task Overview (NEW)
Training data search in object detection.  In the training set search challenge, we aim to search small-scale, yet highly effective training sets from a large-scale data pool such that a competitive target-specific model can be obtained. 

--------------------------------------------------------------------

## Overview (OLD)
This year’s challenge focuses on **Training set optimization for object detection**, where the source and target domains have completely different classes (pedestrian IDs). The particular task is to retrieve the pedestrian instances of the same ID as the query image. This problem is significantly different from previous VisDA challenges, where the source and target domains share some overlapping classes. Moreover, ID matching depends on fine-grained details, making the problem harder than before.

The competition will take place during May -- July 2020, and the top-performing teams will be invited to present their results at the workshop at [ECCV 2020](https://sites.google.com/view/task-cv2020) in September, Glasgow.

## Challenge Data 

[[GoogleDrive]](https://drive.google.com/open?id=18qIbI1XiG2n36qCTS-Te-2XATxiHNVDj) and [[BaiduYun]](https://1drv.ms/u/s!AhjrHmxemkOga91UXOVXsVZJqTg?e=kbE1CC)

![enter image description here](https://github.com/sxzrt/The-PersonX-dataset/raw/master/images/logo1.jpg)



The Region100 benchmark consists of footage captured by 100 static cameras from various regions in the real world.
For videos from each different region, the first 70% is used for model training, while the remaining 30% is designated for validation and testing.

The target domain consists of real-world images from 100 regions in the world. In total, there are 21871 images. We have provided region camera index information for both source and target training sets. 
 - For the training set, it contains 15,369 images.
 - For the test A set, it contains 2,134 images.
 - For the test B set, it contains 4,368 images.


The challenge dataset split is organized as follows: 
```
├── Region100/
(Source dataset collected from synthetic simulator)
│   ├── source_pool/              /* source training images
│       ├── voc_train/                    
│           ├── VOC2007/
|           ├── VOC2012
│       ├── kitti_train/                    
│           ├── 000000.png
|           ├── 000001.png
|           ...
│       ├── coco_train/                    
│           ├── 000000000009.jpg
|           ├── 000000000025.jpg
|           ...
│       ├── cityscapes_train/                    
│           ├── aachen_000000_000019_leftImg8bit.jpg
|           ├── aachen_000001_000019_leftImg8bit.jpg
|           ...
│       ├── bdd_train/                    
│           ├── 0a0a0b1a-7c39d841.jpg
|           ├── 0a0b16e2-93f8c456.jpg
|           ...
│       ├── ade_train/                    
│           ├── ADE_train_00000001.jpg
|           ├── ADE_train_00000002.jpg
|           ...
│       ├── voc_annotation.json
│       ├── kitti_annotation.json
│       ├── coco_annotation.json
│       ├── cityscapes_annotation.json
│       ├── bdd_annotation.json
│       ├── ade_annotation.json 
(Target dataset collected from real world)
│   ├── train/                    /* training set
│       ├── 001/
|           ├── 001_000001.jpg
|           ├── 001_000002.jpg
|           ...
│       ├── 002/
|           ├── 002_000001.jpg
|           ├── 002_000002.jpg
|           ...
│   ├── testA/                    /* validation set
│       ├── 001_000001.jpg
│       ├── 001_000002.jpg
│       ├── 001_000003.jpg
│       ├── 001_000004.jpg
|           ...
│   ├── testB                     /* test set
│       ├── 001_000001.jpg
│       ├── 001_000002.jpg
│       ├── 001_000003.jpg
│       ├── 001_000004.jpg
|           ...
```

By downloading these datasets you agree to the following terms:

### Naming Rule of the images
The first three digits correspond to the region code, and the following six digits correspond to the image number within the region. For example, 001_000001.jpg is the first image captured from the first region camera. "001" is the region code, and "000001" is the image number.


### Terms of Use
- You will use the data only for non-commercial research and educational purposes.
- You will NOT distribute the images.
- The organizers make no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
- You accept full responsibility for your use of the data.

You can download the datasets with the following link: [GoogleDrive](https://drive.google.com/open?id=18qIbI1XiG2n36qCTS-Te-2XATxiHNVDj) and [OneDrive](https://1drv.ms/u/s!AhjrHmxemkOga91UXOVXsVZJqTg?e=kbE1CC).

Moreover, we also provide translated images from SPGAN [2]. SPGAN conducts source-target image translation, such that the translated images follow the distribution of the target. Thus, the Re-ID model trained on the translated images achieves high accuracy on the test set. OneDrive: [PersonX_SPGAN](https://1drv.ms/u/s!AhjrHmxemkOgbIahEx1m49NDuDI?e=i9wE31) or GoogleDrive: [PersonX_SPGAN](https://drive.google.com/open?id=1HEV_EfnLAWU_a5pyeZ12yl5lRCeivDG-).

### Download the Test Set
- The test set is available [here](https://drive.google.com/file/d/12oWSOK1oQVhTNqoVUli70dboXOxPxG41/view?usp=sharing). You need to fill in the [google form](https://forms.gle/9hjryZ5WDUUEZTAX9) to get the password for unzipping files. If you can not open the form, please fill in the [form offline](https://github.com/Simon4Yan/VisDA2020/tree/master/form/) and send it to (weijian.deng@anu.edu.au).

- We respectfully notice that all participating teams should fill in the form above. If you haven't finished, please fill it online or offline. Thanks!

- Please kindly follow the [RULES](http://ai.bu.edu/visda-2020/#rules) and read the [FAQ](http://ai.bu.edu/visda-2020/#faq), thanks!


## Evaluating Your Model
We have provided the evaluation script used by our server so that *you may evaluate your results offline*. You are encouraged to upload your results to the evaluation server to compare your performance with that of other participants.
We will use CodaLab to evaluate submissions and maintain a leaderboard. To register for the evaluation server, please create an account on CodaLab and enter as a participant in the following competition:

[Domain Adaptive Pedestrian Re-identification](https://competitions.codalab.org/competitions/24664)

If you are working as a team, you have the option to register for one account for your team or register multiple accounts under the same team name. If you choose to use one account, please indicate the names of all of the members on your team. This can be modified in the “User Settings” tab. If your team registers for multiple accounts, please do so using the protocol explained by CodaLab here. Regardless of whether you register for one or multiple accounts, your team must adhere to the per-team submission limits (20 entries per day per team during the validation phase). 

The evaluation metrics used to rank the performance of each team will be mean Average Precision (mAP) and Cumulated Matching Characteristics (CMC) curve. **The metrics evaluate the top-100 matches**. 

### Submission Format
Each line of the submitted file contains a list of the top 100 matches from the gallery set for each query, in ascending order of their distance to the query. The delimiter is space. Each match should be represented as the **index** of the gallery image (from 00000 to 24005 for the test set). 

More specifically, the first line of submission file is corresponding to the top 100 matches (represented as indices) of the first query (index=0000); the second line is corresponding to the second query (idex=0001).

- The index of each image in the validation set can be found in [submit-test](https://github.com/Simon4Yan/VisDA2020/tree/master/submit_test).
- Please see a sample submission file [submission-example]( https://github.com/Simon4Yan/VisDA2020/tree/master/submit_test).

### Submitting to the Evaluation Server
[Domain Adaptive Pedestrian Re-identification](https://competitions.codalab.org/competitions/24664)

Once the servers become available, you will be able to submit your results:

- Generate "result.txt".
- Place the result file into a zip file named [team_name]_submission.
  In this step, please directly zip the result file and get "result.zip". You can choose to 
  rename the zip to [team_name]_submission or just submit the "result.zip" for convenience.
- Submit to the CodaLab evaluation server following the instructions below

To submit your zipped result file to the appropriate VisDA challenge click on the “Participate” tab. Select the phase (validation or testing). Select Submit / View Results, fill in the required fields and click “Submit”. A pop-up will prompt you to select the results zip file for upload. After the file is uploaded, the evaluation server will begin processing. This might take some time. To view the status of your submission please select “Refresh Status”. If the status of your submission is “Failed” please check your file is named correctly and has the right format. You may refer to the scoring output and error logs for more details.

After you submit your results to the evaluation server, you can control whether your results are publicly posted to the CodaLab leaderboard. To toggle the public visibility of your results please select either “post to leaderboard” or “remove from leaderboard.”

## Devkit
We provide a simple baseline code ([based on codes [3]](https://github.com/Simon4Yan/feature_learning)). In the devkit, we provide code for reading the challenge datasets and evaluation code.

- The mAP evaluation code in this github evaluates all matches.  And the server evaluates mAP based on top-100 matches (this is commonly used in the community, such as [Aitychallenge](https://www.aicitychallenge.org/2020-data-and-evaluation/)).
Thus, the CMC ranks are identical, while the mAP in the server is higher.  Consider the generality, I provide the code here to evaluate all matches. You could modify the evaluation code to evaluate the top-100 matches, if you want to calculate the same number of mAP with the codalab.

```bash
python learn/train.py
python learn/test.py
```

The baseline performance is, 
|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
| Source Only |26.53 | 14.19 |  [ResNet-50] |
| SPGAN |41.11 | 21.35  |  [ResNet-50] |

## Broader Impact

This competition is featured by learning from synthetic 3D person data. We are not only advancing state-of-the-art technologies in domain adaptation, metric learning and deep neural networks, but importantly aim to reduce system reliance on real-world datasets. While we evaluate our algorithms on real-world data, we have adopted strict measures to take care of the privacy issue. For example, all the faces have been blurred. The participants have signed to comply with our data protection agreement, where we have forbidden the posting or distribution of test images in papers or other public domains. We believe these measures will significantly improve data safety and privacy, while allowing researchers to develop useful technologies.

## Feedback and Help
If you find any bugs please [open an issue](https://github.com/Simon4Yan/VisDA2020/issues/new).

## References

[1] Sun, Xiaoxiao, and Liang Zheng. "Dissecting person re-identification from the viewpoint of viewpoint." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_. 2019.

[2] W. Deng, L. Zheng, Q. Ye, G. Kang, Y. Yang, and J. Jiao. Image-image domain adaptation with preserved self-similarity and domain-dissimilarity for person re-identification. In CVPR, 2018

[3] Lv, Kai, Weijian Deng, Yunzhong Hou, Heming Du, Hao Sheng, Jianbin Jiao, and Liang Zheng. "Vehicle reidentification with the location and time stamp." In _Proc. CVPR Workshops_. 2019.
