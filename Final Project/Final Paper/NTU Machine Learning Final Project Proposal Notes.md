# NTU Machine Learning Final Project Proposal Notes
###### tags: `NTU_ML` `Machine Learning`

## Deep6mAPred: A CNN and Bi-LSTM-based deep learning method for predicting DNA N6-methyladenosine sites across plant species
For this paper, my perspective is this is a little bit trivial to solve the problem. For simplicity speaking, they just change the stacking model structure to a sequence structure. In addition, the result of this paper is exaggerating.
### Comparison
* The result below is the experience on `6mA-rice-LV`(rice) dataset, and this paper method is `Deep6mAPred`. They used 5-fold cross validation on this data(`6mA-rice-LV`). In the original context, they said:
    > The `Deep6mAPred` reached better Sn than three baseline methods (`Deep6mA` , `SNNRice6mA-large` and `Deep6mAPred`), and achieved competitive SP, ACC and MCC in contrast with the `Deep6mA`, which completely outperformed the `SNNRice6mA-large` and MM-6mAPred.

    However, the fun fact is the performance of $Sp, ACC, MCC, AUC$ is not good enough in this dataset. 

    ![](https://imgur.com/vccb3m0.png)
* The result below is for `6mA-rice-chen` dataset. Compared with `Deep6mA`, `Deep6mAPred` increased $Sn$ by 0.1572, ACC by 0.0750, MCC by 0.1436, and AUC of ROC curve by 0.0237, completely superior to the other two methods. The $Sp$ of `Deep6mAPred` is slightly lower than that of `Deep6mA`, but much higher than that of the other two methods.
![](https://imgur.com/ixsSXSK.png)
This result is quite distinguished that can show how special their model is under this another rice data.
* This is ROC curves and PR curves result on `6mA-Fuse-R`(Rosa chinensis) and `6mA-Fuse-F`(Fragaria vesca, a kind of wild strawberry) respectively. In order to show how robust on their model, they try to test different species such as rose and wild strawberry without training, and the result is quite significant that almost similar to rice data.
![](https://imgur.com/eODMhQm.png)
* This is a self-created table that I wanna show the AUC of two curves with different species. 
The original context said:
    > As for the 6mA-Fuse-R, the `Deep6mAPred` outperformed three baseline methods in terms of the AUCs of ROC curves, while in terms of the AUCs of the PR curves it was equivalent to the `Deep6mA` but superior to the `SNNRice6mA-large` and `MM-6mAPred` a bit
    
    Follow the description above, we can know that the result of `6mA-Fuse-R` is better than three baseline methods but without any table or figure to prove that and this is not rigourous enough for this information.
    ![](https://imgur.com/UVaJ7sL.png)
* They also do some ablation experiment to prove that the attention mechanism they choose is quite valid and useful in this project.
We can see that in each experiment of different species, with attention mechanism is generally better than the experiment that without attention.
![](https://imgur.com/3Sj3Tfz.png)

### Other Issue
* Why can wild rose and rice use the same architecture or we can ask how to process input data so that they can be applicable at the same model structure.
* There is no extra explanation for the selected attention mechanism method.


## Ensemble Learning for Brain Age Prediction
The main opinion to this paper is that it's report of the competition they attended. And listed as clear as possible what problems they encountered, what techniques they used etc.
### Comparison
* The * symbol represents a significant reduction in $MAE$ by Ensemble Learning compared to Inception alone ($p\ value < 0.05$)
    * For the objective of minimize MAE, the way of deep learning is better than `BLUP` and `SVM` ($pvalue\ of\ paired\ t-test<3.1e-4$)
    * There was no significant difference in the performance of the deep learning algorithms ($p > 0.027$)
    * In contrast, Ensemble Learning's $MAE=3.46$, there is a significant difference (p=1.3e-4)
    * Taking challenge 2 as an example, the author uses median and mean absolute deviation per site to rescale the prediction. The results show that $MAE$ will increase by one year compared to the original one, but will reduce the bias. The same that ensemble learning has a significant improvement compared to Inception($p=0.010$).
![](https://imgur.com/5jR4hWo.png)

* They also tried to evaluate whether their conclusions depend on the train/test split used in the previous section by performing a 5-fold cross-validation experiment.
    * Within each fold, they found a nominally significant difference in MAE between `BLUP`/`SVM` and `ResNet` ($p < 5.5E−3$)
    * In each fold, the composite age score using linear regression outperformed `Inception V1`'s predictions ($p < 0.0022$). For folds 2 and 3, ensemble learning via random trees significantly outperforms `Inception V1` alone ($p=4.0E−3 and 3.4E−4$)
    * Note that the $MAE$ obtained using Random Forest is very close to the $MAE$ obtained by taking the mean or median score for each person. We cannot conclude that there is a significant difference between **linear model combinations** and **random forests**.
![](https://imgur.com/zyIaGAd.png)
* The low performance of `BLUP`/`SVM` shown above compared to deep learning algorithms motivated the authors to test whether it could be attributed to the input data or the algorithm itself. Therefore, the author retrains `BLUP` and `SVM` <font color="FF0000">(trained on gray matter maps)</font>
    * † Symbols represent: the algorithm trained with gray matter map is significantly **better than** the algorithm trained with surface-based vertices ($p < 0.05/15$).
    * The * symbol indicates: the performance of the algorithm trained on the gray matter image is significantly **lower than** that of `Inception V1` ($p < 0.05/15$)
    * Despite the reduction in MAE, `BLUP-mean` and `SVM` trained on gray matter still performed <font color="FF0000">**worse than**</font> `Inception V1` ($p < 0.0033$), although the difference between `Inception V1` and `BLUP-quantile` became not significant.
![](https://imgur.com/gO86vVb.png)
* The participant is older, the prediction error is larger. → Therefore, the predictor will tends to underestimate the age of older participants and overestimate the age of younger participants.
We did not observe significant associations of prediction errors with gender or location
![](https://imgur.com/a3ugXLy.png)
### Other Issue
* They didn't explain why they used two `6-Layers CNN` to combine and the effect in detailed.
* They also didn't explain the gray/white matter map difference and the properties of these maps in detailed.


## Machine learning workflows
This paper is just like a `Readme` file that wanna teach someone how to use their tool, each technique they used, each problem they encountered, and also which programming package they used etc. as clear as possible. Although the paper should be as clear as possible, but too much unnecessary information is really a waste of time and annoying.
### Comparison
* **Random Forest(RFs)**
    * Vanilla RF(vRF)
        * The ME of vRF was 4.8%, the AUC was 99.9%, and the corresponding BS and LL were 0.32 and 0.78, respectively
        * Platt scaling with LR and `FLR` improves BS and LL by a factor of 2-4, furthermore, `FLR` is better than `LR`
        * MR slightly outperformed Platt's two variants and achieved very low 10th and 9th overall BS (0.073) and LL (0.155) metrics respectively
    * tuned RF(tRF)
        * RF tuned for ME (`tRFME`) showed 10th overall error rate (3.5%) and 4th AUC (99.9%), while it had relatively high BS (0.35) and LL (0.86) similar to `vRF`
        * Both `tRFBS` and `tRFLL` have higher error rates, about 5.5%
        * After calibration with <font color="FF0000">**MR**</font>, almost all versions of `tRF` get the biggest performance improvement

    ![](https://imgur.com/MgjyT0T.png)
* **ELNET**
    * It used 1,000 most variable CpG probes
    * ME ranked 8th, AUC ranked 5th
    * ME (2.7%), BS (0.048) and LL (0.109) and negligibly low AUC (99.9 %)

    ![](https://imgur.com/A2RRIAp.png)
* **SVM**
    * More effective ME = 2.1% (lowest overall) with Platt scaling with Firth regression
    * While simple LR can be more effective to improve BS (second) and LL (fourth) by 8-9 times respectively
    * MR (<font color="FF0000">**`SVM-LK+MR`**</font>) achieves the most comprehensive improvement across all metrics. It reduced BS by a factor of 9.5 and LL by a factor of 11.5, resulting in the second lowest ME (2.1%) and AUC (99.9%), lowest BS (0.039) and lowest LL (0.085)
    
    ![](https://imgur.com/CaeII1m.png)
* **Boost Tree**
    * Boosted model using ME as evaluation metric outperforms model using LL
    * Overall ME of 5.1% and AUC of 99.9%, with the second lowest BS (0.15) and LL (0.43) among the base ML classifiers studied

    ![](https://imgur.com/XXdGN2p.png)
### Other Issue
