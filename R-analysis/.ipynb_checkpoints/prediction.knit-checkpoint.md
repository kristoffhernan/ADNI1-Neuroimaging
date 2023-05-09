---
title: "Predictions"
output: html_document
date: "2023-04-30"
---





```r
library(nnet)
library(ggplot2)
library(MASS)
library(knitr)
library(tidyverse)
```

```
## -- Attaching core tidyverse packages ------------------------ tidyverse 2.0.0 --
## v dplyr     1.1.1     v readr     2.1.4
## v forcats   1.0.0     v stringr   1.5.0
## v lubridate 1.9.2     v tibble    3.2.1
## v purrr     1.0.1     v tidyr     1.3.0
## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
## x dplyr::select() masks MASS::select()
## i Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```



```r
load(file = file.path(root_dir, "R_data", "pca_wm.Rda"))
load(file = file.path(root_dir, "R_data", "pca_gm.Rda"))
load(file = file.path(root_dir, "R_data", "pca_cb.Rda"))
```



## Running models

```r
# not balance weighted
wm_mn_nw <- mn_reg(pca_wm)
gm_mn_nw <- mn_reg(pca_gm)
cb_mn_nw <- mn_reg(pca_cb)

# balance weighted
wm_mn <- mn_reg(pca_wm,weights=TRUE)
gm_mn <- mn_reg(pca_gm,weights=TRUE)
cb_mn <- mn_reg(pca_cb,weights=TRUE)
```



## Summaries

If we look at the coefficient section, we're seeing two models, one modelling the effect of principal components 1:n on the log odds of predicting MCI over CN, while the next model is predicting AD over CN. In multinomial logistic regression, we are estimating coefficients for each level of the response variable (MCI, AD, or CN) relative to a baseline level (CN).

We can interpret the coefficients as say for PC1 and 1. Each additional unit of PC1 decreases the log odds of predicting MCI over CN by 0.001449435. This doesn't tell you much, but its good practice to try and interpret. 


```r
stats_wm_mn <- summary_stats(wm_mn$mod)
stats_wm_mn$coefs[1:5,]
```

```
##                      MCI            AD
## (Intercept)  0.897514509 -0.8358435573
## PC1         -0.001488926 -0.0004086666
## PC2         -0.001843525 -0.0010758174
## PC3          0.005292621  0.0097211992
## PC4         -0.001183938  0.0004192962
```

```r
stats_gm_mn <- summary_stats(gm_mn$mod)
stats_gm_mn$coefs[1:5,]
```

```
##                       MCI            AD
## (Intercept)  0.6463250234 -0.5349365963
## PC1          0.0006229977  0.0004933212
## PC2         -0.0013413063 -0.0024110653
## PC3          0.0045772788  0.0073832443
## PC4          0.0055115438  0.0027044121
```

```r
stats_cb_mn <- summary_stats(cb_mn$mod)
stats_cb_mn$coefs[1:5,]
```

```
##                      MCI            AD
## (Intercept)  0.191166565 -0.2052170939
## PC1         -0.002743238  0.0014026933
## PC2          0.007545769  0.0007824543
## PC3         -0.004382656  0.0005000055
## PC4          0.003664760  0.0020831190
```



## Wald Statistics

Wald Test

$$\hat{\beta}_{MLE} - \beta \sim N(0, I^{-1}_n) \text{ as } n \rightarrow \infty$$

Test Statistic

$$
\frac{|\hat{\beta}_{MLE} - 0|}{\sqrt{1/I_n}} > z_{1-\frac{\alpha}{2}}
$$

We can use the Wald Test to test the significance of the coefficients in our model.

$$H_0: \beta_j = 0 \\ H_1: \beta_j \neq 0$$

Here, a large value would indicate the coefficient being significantly different from 0.


```r
# wald statistics from each segment
stats_wm_mn$wald_stats[1:5,]
```

```
##                   MCI          AD
## (Intercept) 1.2441141 0.480335707
## PC1         1.0091079 0.070316744
## PC2         0.8956024 0.098107740
## PC3         1.7197583 0.248963724
## PC4         0.4236131 0.004212392
```

```r
stats_gm_mn$wald_stats[1:5,]
```

```
##                   MCI         AD
## (Intercept) 0.6128240 0.33583517
## PC1         0.2001834 0.09142254
## PC2         0.3480175 0.37660542
## PC3         0.6135336 0.57412132
## PC4         0.9441592 0.08368866
```

```r
stats_cb_mn$wald_stats[1:5,]
```

```
##                   MCI         AD
## (Intercept) 0.1009102 0.10931465
## PC1         0.4201646 0.30006684
## PC2         0.6903313 0.10759029
## PC3         0.2360622 0.03142193
## PC4         0.7188717 0.27514854
```

```r
# selecting PCs if one of the wald statistics is greater than 2
stats_wm_mn$wald_results
```

```
## [1] MCI AD 
## <0 rows> (or 0-length row.names)
```

```r
stats_gm_mn$wald_results
```

```
## [1] MCI AD 
## <0 rows> (or 0-length row.names)
```

```r
stats_cb_mn$wald_results
```

```
## [1] MCI AD 
## <0 rows> (or 0-length row.names)
```

```r
# significant coefficients
num_sig_coefs_df <- data.frame(wm_mn = stats_wm_mn$num_sig_coefs, 
                                gm_mn = stats_gm_mn$num_sig_coefs, 
                                cb_mn = stats_cb_mn$num_sig_coefs)
colnames(num_sig_coefs_df) <- c('wm_mn', 'gm_mn', 'cb_mn')
kable(num_sig_coefs_df, 
      caption='Number of Significant Coefficients for Each Segment (Multinoimal Models)')
```



Table: Number of Significant Coefficients for Each Segment (Multinoimal Models)

| wm_mn| gm_mn| cb_mn|
|-----:|-----:|-----:|
|     0|     0|     0|




## confusion matrix

```r
wm_mn$cm
```

```
##             y_preds
##              CN_pred MCI_pred AD_pred
##   CN_actual       26        4       0
##   MCI_actual       4       37       0
##   AD_actual        2        3      11
```

```r
gm_mn$cm
```

```
##             y_preds
##              CN_pred MCI_pred AD_pred
##   CN_actual       24        5       1
##   MCI_actual       5       34       2
##   AD_actual        0        3      13
```

```r
cb_mn$cm
```

```
##             y_preds
##              CN_pred MCI_pred AD_pred
##   CN_actual       27        0       3
##   MCI_actual       4       33       4
##   AD_actual        1        1      14
```

One problem with accuracy is that if we have unbalanced data, it can be deceiving. Imagine you have two classes apples (99% of the data) and oranges (1% of the data). We might have a 99% accuracy even though we wrongly predicted the orange. This would be wrong. In our case, we have examples like with our white matter multinomial model where we have an 86% accuracy, but if we split it up by each group, the accracy is high because we correctly classified 40/41 MCI. However, we only correctly classified AD 8/16. 

Initially, we faced challenges in getting accurate results due to unbalanced splits, as our test data would sometimes consist mostly of MCI and CN samples, with very few AD samples. To solve this we split the data proportionally and got better results like the above. 

To explain, in our full dataset proportionally there are 151/434 (0.348) CN, 206/434 (.475) MCI, and 77/434 (0.177) AD. Originally we then did an 80/20 split, where 80% or 347 of the 434 samples are set aside for the train and 87 are set aside for the test. However, this time, of those 347 sample, 34.8% will be CN, 47.5% will be MCI and 17.7% will be AD.

This misclassification can cause further problems as having a false negative (failing to classifying someone as AD and instead saying theyre CN) can be more fatal and thus should have a higher weight. 




## LDA

```r
wm_lda <- lda_reg(pca_wm)
gm_lda <- lda_reg(pca_gm)
cb_lda <- lda_reg(pca_cb)
```



```r
wm_lda$cm
```

```
##             y_preds
##              CN_pred MCI_pred AD_pred
##   CN_actual       27        2       1
##   MCI_actual       4       37       0
##   AD_actual        0        3      13
```

```r
gm_lda$cm
```

```
##             y_preds
##              CN_pred MCI_pred AD_pred
##   CN_actual       29        0       1
##   MCI_actual       4       37       0
##   AD_actual        1        2      13
```

```r
cb_lda$cm
```

```
##             y_preds
##              CN_pred MCI_pred AD_pred
##   CN_actual       29        0       1
##   MCI_actual       3       37       1
##   AD_actual        1        3      12
```


## LDA Assumptions

#### Homoskedasticity Among Classes

Bartlett's K-Squared Test

$H_0: \sigma_{CN} = \sigma_{MCI} = \sigma_{AD}$ Variance is the same across all classes

$H_1: \sigma_{CN} \neq \sigma_{MCI} \neq \sigma_{AD}$ Variance isn't the same across all classes

$\alpha = 0.05$


```r
class_data <- split(pca_wm$X_train_pca, pca_wm$y_train)

# Calculate variances of each column in each class
variances <- matrix(NA, nrow = ncol(pca_wm$X_train_pca), ncol = length(class_data), dimnames = list(NULL,c("CN", "MCI", "AD")))

for (i in 1:length(class_data)) {
  vars <- sapply(class_data[[i]], var)
  variances[,i] <- vars
}

var_df <- data.frame(variances) %>%
            pivot_longer(everything(), names_to='Group', values_to='Var')
head(var_df)
```

```
## # A tibble: 6 x 2
##   Group     Var
##   <chr>   <dbl>
## 1 CN    233160.
## 2 MCI   169768.
## 3 AD    234514.
## 4 CN     82975.
## 5 MCI   100754.
## 6 AD     63098.
```



```r
# define a function to perform the KS test for normality on each column of a data frame
ks_test <- function(x) {
  ks.test(x, "pnorm")$p.value
}

# apply the KS test to each column of each subset
results <- lapply(class_data, function(subset) {
  lapply(subset, ks_test)
})


ks_df <- data.frame(cbind(results$`0`, results$`1`, results$`2`)) 
colnames(ks_df) <- c('CN', 'MCI', 'AD')
head(ks_df)
```

```
##     CN MCI           AD
## PC1  0   0 1.665335e-15
## PC2  0   0 2.442491e-15
## PC3  0   0 1.665335e-15
## PC4  0   0 1.665335e-15
## PC5  0   0 1.665335e-15
## PC6  0   0 1.665335e-15
```

```r
normal_pcs <- ks_df %>%
    filter(any(. >= 0.05))
normal_pcs 
```

```
## [1] CN  MCI AD 
## <0 rows> (or 0-length row.names)
```

```r
non_normal_pcs <- ks_df %>% 
      filter(any(. < 0.05))
    
non_normal_pcs %>% summarise(n=n())
```

```
##     n
## 1 160
```
We reject the null for each feature for all classes. LDA assumptions are not met.



## Plot method comparison

```r
method <- c(rep("mn", 3), rep("lda", 3))
segment <- rep(c("wm", "gm", "cb"), 2)
accuracy <- c(wm_mn$accuracy, gm_mn$accuracy, cb_mn$accuracy,
              wm_lda$accuracy, gm_lda$accuracy, cb_lda$accuracy)

# combine the vectors into a data frame
df <- data.frame(method = method, segment = segment, accuracy = accuracy)

kable(df, format = "markdown")
```



|method |segment |  accuracy|
|:------|:-------|---------:|
|mn     |wm      | 0.8505747|
|mn     |gm      | 0.8160920|
|mn     |cb      | 0.8505747|
|lda    |wm      | 0.8850575|
|lda    |gm      | 0.9080460|
|lda    |cb      | 0.8965517|

```r
ggplot(df, aes(x = segment, y = accuracy, fill = method)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Accuracy by Method and Segment",
       x = "Segment", y = "Accuracy",
       fill = "Method") +
  ylim(0, 1) + 
  theme_minimal()
```

![](prediction_files/figure-latex/unnamed-chunk-11-1.pdf)<!-- --> 



## True Group Percentage

```r
df <- data.frame(wm_mn = calc_indiv_acc(wm_mn),
           gm_mn = calc_indiv_acc(gm_mn),
           cb_mn = calc_indiv_acc(cb_mn),
           wm_mn_nw = calc_indiv_acc(wm_mn_nw),
           gm_mn_nw = calc_indiv_acc(gm_mn_nw),
           cb_mn_nw = calc_indiv_acc(cb_mn_nw),
           wm_lda = calc_indiv_acc(wm_lda),
           gm_lda = calc_indiv_acc(gm_lda),
           cb_lda = calc_indiv_acc(cb_lda)
          )
rownames(df) <- c("CN","MCI","AD")
kable(t(df), caption='Group Accuracy % Per Model & Segment')
```



Table: Group Accuracy % Per Model & Segment

|         |    CN|   MCI|    AD|
|:--------|-----:|-----:|-----:|
|wm_mn    | 0.867| 0.902| 0.688|
|gm_mn    | 0.800| 0.829| 0.812|
|cb_mn    | 0.900| 0.805| 0.875|
|wm_mn_nw | 0.900| 0.976| 0.500|
|gm_mn_nw | 0.900| 0.829| 0.688|
|cb_mn_nw | 0.900| 0.756| 0.688|
|wm_lda   | 0.900| 0.902| 0.812|
|gm_lda   | 0.967| 0.902| 0.812|
|cb_lda   | 0.967| 0.902| 0.750|
