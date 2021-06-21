# StockPrediction_CapstoneDA2021-1
------------------------------------
해당 Repository는 경희대학교 2021년 1학기 데이터분석캡스톤 디자인 수업의 일환으로 만들어졌습니다. (제작중)

* 참여자: 신휘명

## Overview
- 개요: 지난 1년간 개인의 주식투자에 대한 관심이 폭발적으로 늘었다. 뉴스기사에 따르면 한국 주식 투자자수는 재작년 2019년(약 600만명)과 비교하여 300만명가량이 증가하여 900만명을 넘어섰다. 신규 투자자의 절반가량은 20~30대였다. 하지만 개인 투자자는 소위 ‘개미’라고 불려 기관 등의 대형 투자자들에 비해 영향력도, 정보도 크게 밀린다. 고급정보의 부족으로 개미는 주식시장에서 투자성공을 장담하기 어렵다. 이러한 정보의 격차를 줄이기 위해 주식시장의 동향을 읽고 수많은 기업정보를 자동으로 분석해주는 기능이 필요하다.
- 주요내용: Python에서 제공하는 라이브러리인 FinanceDataReader에서 제공하는 주식차트 데이터를 활용하여 해당 종목의 다음 주가 상승여부를 예측하는 딥러닝 모델을 설계하고 훈련하였다. 제공하는 주식 데이터의 종류와 방법을 바꿔가며 성능의 변화를 측정하였다. 주가 예측에 도움이 되는 지표가 무엇일지 다양한 실험을 통해 유추해보았으며 웹 크롤링을 이용하여 최신의 주가 데이터를 지속적으로 학습할 수 있도록 하였다.
- 목표: 개발된 딥러닝 모델이 예측한 정보(n일 이후의 차트 예측 혹은 상승 여부 등)을 바탕으로, 실전 주식투자에서 수익을 거두는 것이 최종 목표이다.
- 방법: Python 프로그래밍 언어와 FinanceDataReader 라이브러리를 활용한다. 딥 러닝 모델의 설계는 tensorflow (LSTM류의 sequence data를 사용하는 딥러닝 모델 설계에 사용)와 pytorch(Binary Classification 모델 설계에 사용)를, 데이터 처리에는 pandas와 numpy를 활용에 알맞게 사용하였다. FinanceDataReader로부터 주식 차트 데이터를 불러와 적절한 전처리를 한 후, 설계한 딥러닝 모델에 넣어 다음 차트의 동향을 예측하도록 하였다. 지속적인 학습을 위한 최신 데이터는 Investing.com을 크롤링하여 얻어내었다.

Main code
--------------------
Load & Proces Data (각 종목의 종가)
--------------------
* FinancialDataLoader를 이용하여 코스피 종목의 종가를 불러올 수 있다.
split_iter를 이용하여 서로 다른 시작점과 끝점을 가진 split_iter개의 sequantial 데이터가 만들어진다. 
```python
import FinanceDataReader as fdr

read_lines = np.flip(df_kospi.to_numpy(), axis=0)[:100]
...

for line in np.flip(read_lines, axis=0):
 try:
   df = fdr.DataReader(line[0], start_date, end_date)
  df_ratio = df.iloc[:, 3].astype('float32')
  df_log1 = pd.DataFrame(df_ratio)
  df_ratios = np.append(df_ratios, df_ratio.to_numpy())

  for j in range(0,split_iter):
      split_point_start = j * max_test_size
      split_point_end = (split_iter - j + 1) * max_test_size
      df_train1 = df_log1.iloc[-max_train_size+split_point_start:-split_point_end]
      df_test1 = df_log1.iloc[-split_point_end:-split_point_end+max_test_size]
```

Load & Proces Data (추가지수)
--------------------
* FinancialDataLoader를 이용하여 종목 뿐 아니라 주식시장에 영향을 미치는 각종 지수의 데이터를 불러올 수 있다.
10가지 추가 지수를 불러와 이전에 불러온 각 종목의 종가 데이터와 날짜가 같은 것끼리 결합(concatenate)한다. 날짜 쌍이 안 맞는 데이터는 버려진다(.dropna).
```python

df2 = fdr.DataReader('KS11', start_date, end_date)

df_ratio2 = df2.iloc[:, 0:1].astype('float32').fillna(0)
df_log2 = pd.DataFrame(df_ratio2)


df_dict = {
    0 : fdr.DataReader('IXIC', start_date, end_date),#나스닥
    1 : fdr.DataReader('KQ11', start_date, end_date),#코스닥
    2 : fdr.DataReader('USD/KRW', start_date, end_date),#달러/원
    3 : fdr.DataReader('KS50', start_date, end_date),#코스피50
    4 : fdr.DataReader('KS100', start_date, end_date),#코스피100
    5 : fdr.DataReader('KS200', start_date, end_date),#코스피200
    6 : fdr.DataReader('NG', start_date, end_date),#천연가스 선물
    7 : fdr.DataReader('ZG', start_date, end_date),#금 선물
    8 : fdr.DataReader('VCB', start_date, end_date),#베트남무역은행
    9 : fdr.DataReader('US1MT=X', start_date, end_date),#미국채권1개월수익률
}

for i in range(len(df_dict)):
  extra_df = df_dict[i]
  df_ratio_extra = extra_df.iloc[:, 0:1].astype('float32').fillna(0) #((extra_df.iloc[:, 0:1].astype('float32') - extra_df.iloc[:, 0:1].shift().astype('float32')) / extra_df.iloc[:, 0:1].shift().astype('float32')).fillna(0)
  df_log_extra = pd.DataFrame(df_ratio_extra)

  df_log2 = pd.concat([df_log2, df_log_extra],axis=1)

  df_train2 = df_log2.iloc[:]
  df_test2 = df_log2.iloc[:]

  df_train =pd.concat([df_train1, df_train2],axis=1).dropna(axis=0)[-min_train_size:]
  df_test = pd.concat([df_test1, df_test2],axis=1).dropna(axis=0)
```

등락률 계산
--------------------
* 합쳐진 학습데이터를 하나씩 읽어 전날대비 등락률로 데이터를 변조하였다.
```python
df_train_ = np.array([])
previous_train = np.zeros(df_train.shape[1])
for num, i in enumerate(df_train.to_numpy()):#[::sample_step]):
    if num == 0:
        df_train_ = np.expand_dims(previous_train, axis=0) 
    else:
        if (previous_train == 0).any():
          print(previous_train)
        new_item = (i - previous_train) / previous_train
        df_train_ = np.append(df_train_, np.expand_dims(new_item, axis=0), axis=0)
    previous_train = i
```

Classification용 타겟 데이터 생성
--------------------
* max_test_size(11로 설정) 일 이후의 종가의 변화율을 계산하여 그것이 2% 이상이면 '상승', -2% 이하면 '하락', 둘 다 아닌 것은 '유지'로 분류하였다.
```python
df_test = df_test.to_numpy()
df_test = np.array([(df_test[-1] - df_test[0])/ df_test[0] >= 0.02, (df_test[-1] - df_test[0])/ df_test[0] < -0.02])
df_test = np.append(df_test, np.expand_dims(np.logical_not(df_test[0]) * np.logical_not(df_test[1]), axis=0), axis=0)
```

모델 변경
--------------------
* 시퀀스 데이터를 처리하기 위해 기존의 2D Conv를 모두 1D Conv로 교체하였다. 
```python
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
```

정확도 측정
--------------------
* 실제 수익성을 알아보기 위해 accurcay를 포함한 세가지 지수(Precision, Crucial-Fail, Soso-Fail)를 계산하였다.
또한, Thresholding을 통해 최적의 precision을 갖는 threshold를 선택하여 사용할 수 있다. (utils.py)

```python
def accuracy(output, target, threshold=0.5):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    output = output > threshold
    target = target.squeeze()
    correct = (target == output)
    #print("C: ", correct)
    correct = torch.all(correct, dim=1)

    predicted_positive_indexes = (output[:,0].nonzero(as_tuple=True)[0])
    true_positive = (target[:,0])[predicted_positive_indexes]

    crucial_fail = (target[:,1])[predicted_positive_indexes]
    soso_fail = (target[:,2])[predicted_positive_indexes]
    
    return correct.sum() / output.shape[0], true_positive.sum() / predicted_positive_indexes.shape[0], crucial_fail.sum() / predicted_positive_indexes.shape[0], soso_fail.sum() / predicted_positive_indexes.shape[0]
    
```

Tables and Graphs
------------------------------------------
* LSTM을 사용하여 주가를 예측한 결과
![image](https://user-images.githubusercontent.com/40812418/122758082-57543180-d2d3-11eb-870b-6d2255fd6b07.png)

* 1D ResNet을 사용하여 주가를 예측한 결과(Accuracy/Precision/Crucial-Fail/Soso-Fail)
![image](https://user-images.githubusercontent.com/40812418/122758439-c2056d00-d2d3-11eb-9c5f-8829488c0483.png)

GradCAM
--------------------
* 각 추가 지수의 기여도를 정량적으로 알아보기 위하여 유명한 딥러닝 분석 기법인 GradCAM을 사용하였다.
각 지수가 생성하는 gradient를 그래프로 나타낼 수 있었으며 그것의 합을 계산하여 그 합이 큰 순으로 순위를 매길 수 있었다.
![image](https://user-images.githubusercontent.com/40812418/122761653-70f77800-d2d7-11eb-9beb-896a30695220.png)


지속적 학습(준비 중)
--------------------
* 크롤링을 통해 최신 데이터를 읽은 후 그 데이터를 사용하여 모델을 fine-tuning하는 것으로 모델의 성능을 유지하고, 향상시킬 수 있다.

## Conclusion
* 앞서 구해낸 세 가지 지수를 이용하여 예상 수익률을 계산할 수 있다. 
> 투입 금액 : a
> 하락 / 유지 / 상승 구분 기준 : ~-2% / -2% ~ +2% / +2%~
> 상승은 무조건 2% 상승, 하락은 무조건 2% 하락, 유지는 +-0%라 가정
> 투자는 상승이라 예측한 종목 중에서 무작위로 일정 수를 구매하는 방법을 채택.
> 이 중 precision 확률 만큼의 종목이 올라 수익을 얻을 것이고, crucial fail 확률 만큼의 종목이 떨어져 손실을 볼
> 것이며, soso fail 확률에 해당하는 종목은 변동이 없을 것이다.

* 위와 같이 가정하면 수익률 계산법은 아래와 같다.
> 투자 후 주식 가치 : (Precision x 1.02 + Crucial_Fail x 0.98 + Soso_Fail x 1) x a
> 수익 : (Precision x 1.02 + Crucial_Fail x 0.98 + Soso_Fail x 1) x a – a – 수수료

* 이에 가장 좋은 실험 결과로 얻어낸 값을 대입하면 아래와 같다.
> a = 100만원 Precision = 0.59, Crucial_Fail = 0.14 Soso_Fail = 0.27 이라 가정
> 투자 후 주식 가치 : (0.59 x 1.02 + 0.14 x 0.98 + 0.27 x 1) x 100만 = 100.9만
> 수익: 9천원 (+0.9%) (수수료 제외)
> 11일 후의 주가를 예측하므로 약 2주마다 0.9%의 수익을 낼 수 있다.
> 한 달에 두 번 주식 투자를 한다고 가정하면 1년 동안 pow(1.09, 24) = 1.24 배 (+24%)의 이익을 낼 수 있다.

* 이는 일반적인 저가 예금의 수익률(약 1년에 2%)과 비교해봤을 때 매우 큰 수치로, 충분히 주식 투자에 사용할 가치가 있는 정확도를 보인다고 할 수 있다.
* 그러나 최근 날짜의 주가가 계속 업데이트되며 학습된 데이터로부터 날짜가 멀어질 경우 점점 정확도가 하락하기 때문에 지속적 학습을 통해 정확도를 유지하는 것이 중요하다.


## Contributions
1. FinancialDatareader를 이용하여 주식 종목 불러오기/프로세싱/딥러닝 학습
2. 일반적인 LSTM 기반 방법이 아닌 Classification 방법을 사용하여 등락률을 맞추는 문제로 간략화, 실질적 정확도를 높임
3. 종목 예측에 도움이 되는 추가 지수를 결합하여 정확도 향상
4. 정확도 계산을 위한 세가지 추가 metric을 도입하여 실제 수익률 예측
5. GradCAM을 이용하여 각 추가 지수의 기여도 분석
6. 지속적 학습을 통한 예측 정확도 유지 및 향상 (준비중)
