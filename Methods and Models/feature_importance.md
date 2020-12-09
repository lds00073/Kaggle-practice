### Permutation importance

- 학습된 모델이 필요.
-  feature을 돌아가면서 셔플 시킴--> 모델의 정확도가 많이 떨어질수록 중요 변수
- 계산이 빠름/ 사용범위가 넓음/ 이해하기 쉬움

#### 사용 library : eli5

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, scoring='f1', random_state=42).fit(X_val,y_val)
eli5.show_weights(perm, top = 80, feature_names= X_val.columns.tolist())
																	### feature_names 를 지정해야하네...?
## 출처: https://hong-yp-ml-records.tistory.com/51
```

- weight 이 음수가 나오면, 해당 feature 은 shuffle 된 데이터가 실제 데이터로 예측할 때보다, 정확도가 높은 것. 
- 변수를 제거할 필요가 있을 때 사용 가능



#### importance 계산방법

![image-20201027105911343](/Users/이루다/Library/Application Support/typora-user-images/image-20201027105911343.png)

- feature마다 shuffle 하고 모델학습하는 것을 K 번 진행.
- 원래의 스코어 - Corrupted data로 학습했을 때 스코어의 평균  = importance



# 

##  # f1 score



#### True positive 

: 올바른 것(positive)을 올바르게 예측(True)

#### True negative for A

: 거절해야할 것을(Negative)를 잘 거절한 것 (True ), i.e., A라 하면 안되는 것을, A 가 아니라고 함

#### False Positive for A

: 올바른 것(Positive) 한 것을 잘못 거절/ 다른 것으로 예측(False) (ex. A를 B,C,D로 예측한 것)

#### False Negative for A

:  거절해야할 것을 거절하지 않고 잘못 맞다고 함.



#### Performance Measure 1. Accuracy

Accuracy 

-  $\sum(True Positive)\over \sum{Confusion Matrix} $
-  works well on balanced data



#### Performace Measure 2. f1-score

##### Recall : input class에 대해서, 분류기가 어떤 class 로 예측하는지에 대한 척도(Confusion matrix의 행방향)

##### Precision: 예측한 값들 중에서 제대로 예측했는지에 대한 척도(Confusion matrix의 칼럼 방향)

##### f1-score 는 Recall과 Precision을 이용하여 조화평균(Harmonic mean)을 이용한 score이다.

#####  $ f1 score = 2\times {Precision * Recall \over{Precision+Recall}} $

Imbalanced data --> some fold may not contain a positive sample so that TP = FN = 0











