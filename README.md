# Image_Text_Retrieval

## Motivation
Cross-modal learning involves information obtained from more than one modality. Fashion clothing industry is one such field where product retrieval based on multiple modalities such as image and text has become important. In the online fashion industry, being able to search for a product that matches either an image query or a text query is in high demand.

![alt text](../assets/retrieval.png)

## Our Work
In this work, we implement different cross-modal learning schemes such as Siamese Network, Correlational Network and Deep Cross-Modal Projection Learning model and study their performance. We also propose a modified Deep Cross-Modal Projection Learning model that uses a different image feature extractor. We evaluate the model’s performance on image-text retrieval on a fashion clothing dataset.

### Research Paper 
We have written a research paper in the ACM format detailing all aspects of our work. 
Link to paper: [Click_here](https://drive.google.com/file/d/1IhMKgzaUeqUg77lBVWRy9jtksuY6vYnu/view?usp=sharing)

### Presentation Slides
Link to the slides for final project presentation: [Click_here](https://drive.google.com/file/d/1DMOiNq-IsfX6He2XszFinAchVx_Cy4BU/view?usp=sharing)

### Videos

Link to video explaining model architectures and results:
[Click_here](https://drive.google.com/file/d/19Q8W4H52BMJBjPYj_yD0B2o9nXVaooAX/view?usp=sharing)


Link to video showing the webapp demo:
[Click_here](https://drive.google.com/file/d/1FuU6j674pDJKMWXJhVhkiLFbXbJL5eCC/view?usp=sharing)

## Instructions to run the code

The repository contains 3 folders each of which contains the source code for different model architectures we experimented with. Specifically these are:
* Deep CMPL model
* Siamese Network
* Correlational Network

Each of the folders contains a dedicated Readme detailing the instructions to run the source code for each model. The source code is well commented and readable.

<br>

# Theoretical Details

## Model Architectures

### Siamese Network
Siamese Network is a neural network architecture that contains two or more identical sub-networks having the same weights and parameters. It is commonly used to find the similarity of the inputs by comparing its feature vector outputs. We implemented a two-branch neural network inspired from Siamese Network architecture and used a contrastive loss function for our task.

<br>

#### *Network Architecture*
![alt text](../assets/siamese.png)

<br>

### Correlational Network
Correlational Network is an autoencoder based approach that explicitly maximises correlation between image and text embedded vectors in addition to minimising the error of reconstructing thetwo views(image and text). This model also has two branches -one for images and one for text, but at the same time it also has anencoder and decoder.

<br>

#### *Network Architecture*
![alt text](../assets/correlational.png)

<br>

### DEEP CMPL Network
Cross-Modal Projection Learning includes Cross-Modal Projection Matching (CMPM) loss for learning discriminative image-text embeddings. This novel image-text matching loss minimizes the relative entropy between the projection distributions and thenormalized matching distributions. 

#### Modified Deep CMPL
We modified the Deep Cross-Modal Projection Learning modelby using the EfficientNet architecture instead of MobileNet as the image feature extractor. EfficientNet is a recently pro-posed convolutional neural architecture which outperforms other state-of-the-art convolutional neural networks both in terms of efficiency and accuracy.

<br>

#### *Network Architecture*
![alt text](../assets/deepcmpl.png)

<br>

## Experimentations

We experimented different combinations of text and image feature extractors for learning common image-text embeddings. The tested combinations include:
* Experiment 1: Siamese Network
* Experiment 2: Correlational Network
* Experiment 3: Deep CMPL with MobileNet
* Experiment 4: Deep CMPL with EfficientNet on Indian Fashion
* Experiment 5: Deep CMPL with EfficientNet on DeepFashion

<br>
The metrics obtained from these experiments are as follows:
<br>

![alt text](../assets/i2t.PNG)

![alt text](../assets/t2i.PNG)


## References
[1] Marco Basaldella, Elisa Antolli, Giuseppe Serra, and Carlo Tasso. 2018. Bidirec-
tional LSTM Recurrent Neural Network for Keyphrase Extraction. , 180-187 pages.

https://doi.org/10.1007/978-3-319-73165-0_18

[2] Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Säckinger, and Roopak Shah.
1993. Signature Verification Using a "Siamese" Time Delay Neural Network. In
Proceedings of the 6th International Conference on Neural Information Processing
Systems (Denver, Colorado) (NIPS’93). Morgan Kaufmann Publishers Inc., San
Francisco, CA, USA, 737–744.

[3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Im-
ageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on

Computer Vision and Pattern Recognition. 248–255. https://doi.org/10.1109/CVPR.
2009.5206848

[4] Dehong Gao, Linbo Jin, Ben Chen, Minghui Qiu, Peng Li, Yi Wei, Yi Hu, and Hao
Wang. 2020. FashionBERT: Text and Image Matching with Adaptive Loss for
Cross-modal Retrieval. arXiv:2005.09801 [cs.IR]

[5] Richard Zemel Gregory Koch and Ruslan Salakhutdinov. 2015. Siamese Neural
Networks for One-shot Image Recognition, ICML 2015.

[6] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Wei-
jun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. 2017. Mo-
bileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.

arXiv:1704.04861 [cs.CV]

[7] Harshil Jain, Rohit Patil, Utsav Jethva, Ronak Kaoshik, Shaurya Agarawal, Ritik
Dutta, and Nipun Batra. 2021. Generative Fashion for Indian Clothing. In 8th
ACM IKDD CODS and 26th COMAD (Bangalore, India) (CODS COMAD 2021).
Association for Computing Machinery, New York, NY, USA, 415. https://doi.
org/10.1145/3430984.3431057

[8] Shaoqing Ren Kaiming He, Xiangyu Zhang and Jian Sun. 2015. Deep Residual
Learning for Image Recognition.

[9] Furkan Kınlı, Barış Özcan, and Furkan Kıraç. 2019. Fashion Image Retrieval with
Capsule Networks. arXiv:1908.09943 [cs.CV]

[10] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. 2016. DeepFash-
ion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations.

In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition
(CVPR).

[11] Yin Li Liwei Wang and Svetlana Lazebnik. 2016. Learning deep structure-preserving
image-text embeddings, CVPR 2016.

[12] Yin Li Liwei Wang and Svetlana Lazebnik. 2017. Learning two-branch neural
networks for image-text matching tasks.

[13] Luke Melas-Kyriazi. [n.d.]. EfficientNet PyTorch. Retrieved March, 2021 from
https://github.com/lukemelas/EfficientNet-PyTorch

[14] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient
Estimation of Word Representations in Vector Space. arXiv:1301.3781 [cs.CL]

[15] A. Rubio, LongLong Yu, E. Simo-Serra, and F. Moreno-Noguer. 2017. Multi-
modal joint embedding for fashion product retrieval. In 2017 IEEE International

Conference on Image Processing (ICIP). 400–404. https://doi.org/10.1109/ICIP.2017.
8296311

[16] Hugo Larochelle Sarath Chandar, Mitesh M Khapra and Balaraman Ravindran.
2015. Correlational Neural Networks.

[17] Mingxing Tan and Quoc V. Le. 2020. EfficientNet: Rethinking Model Scaling for
Convolutional Neural Networks. arXiv:1905.11946 [cs.LG]

[18] Mikolaj Wieczorek, Barbara Rychalska, and Jacek Dabrowski. 2021. On the Unrea-
sonable Effectiveness of Centroids in Image Retrieval. arXiv:2104.13643 [cs.CV]

[19] Fei Yan and Krystian Mikolajczyk. 2015. Deep correlation for matching images
and text, CVPR 2015.

[20] Erwin M. Bakker Yu Liu, Yanming Guo and Michael S. Lew. 2017. Learning a
Recurrent Residual Fusion Network for Multimodal Matching, ICCV 2017.

[21] Ying Zhang and Huchuan Lu. 2018. Deep Cross-Modal Projection Learning for
Image-Text Matching, ECCV 2018.






