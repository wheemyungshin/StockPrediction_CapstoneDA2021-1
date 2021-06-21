# StockPrediction_CapstoneDA2021-1
------------------------------------
해당 Repository는 경희대학교 2021년 1학기 데이터분석캡스톤 디자인 수업의 일환으로 만들어졌습니다. (제작중)

* 참여자: 신휘명

## Overview
- 개요: 지난 1년간 개인의 주식투자에 대한 관심이 폭발적으로 늘었다. 뉴스기사에 따르면 한국 주식 투자자수는 재작년 2019년(약 600만명)과 비교하여 300만명가량이 증가하여 900만명을 넘어섰다. 신규 투자자의 절반가량은 20~30대였다. 하지만 개인 투자자는 소위 ‘개미’라고 불려 기관 등의 대형 투자자들에 비해 영향력도, 정보도 크게 밀린다. 고급정보의 부족으로 개미는 주식시장에서 투자성공을 장담하기 어렵다. 이러한 정보의 격차를 줄이기 위해 주식시장의 동향을 읽고 수많은 기업정보를 자동으로 분석해주는 기능이 필요하다.
- 주요내용: Python에서 제공하는 라이브러리인 FinanceDataReader에서 제공하는 주식차트 데이터를 활용하여 해당 종목의 다음 주가 상승여부를 예측하는 딥러닝 모델을 설계하고 훈련하였다. 제공하는 주식 데이터의 종류와 방법을 바꿔가며 성능의 변화를 측정하였다. 주가 예측에 도움이 되는 지표가 무엇일지 다양한 실험을 통해 유추해보았으며 웹 크롤링을 이용하여 최신의 주가 데이터를 지속적으로 학습할 수 있도록 하였다.
- 목표: 개발된 딥러닝 모델이 예측한 정보(n일 이후의 차트 예측 혹은 상승 여부 등)을 바탕으로, 실전 주식투자에서 수익을 거두는 것이 최종 목표이다.
- 방법: Python 프로그래밍 언어와 FinanceDataReader 라이브러리를 활용한다. 딥 러닝 모델의 설계는 tensorflow (LSTM류의 sequence data를 사용하는 딥러닝 모델 설계에 사용)와 pytorch(Binary Classification 모델 설계에 사용)를, 데이터 처리에는 pandas와 numpy를 활용에 알맞게 사용하였다. FinanceDataReader로부터 주식 차트 데이터를 불러와 적절한 전처리를 한 후, 설계한 딥러닝 모델에 넣어 다음 차트의 동향을 예측하도록 하였다. 지속적인 학습을 위한 최신 데이터는 Investing.com을 크롤링하여 얻어내었다.

## Schedule
| Contents | March | April |  May  | June  |   Progress   |
|----------|-------|-------|-------|-------|--------------|
|  Topic1  |       |       |       |       |     Link1    |
|  Topic2  |       |       |       |       |     Link2    |

## Results
* Main code

** load_data
FinancialDataLoader를 이용하여 코스피 종목의 종가를 불러올 수 있다.
split_iter를 이용하여 서로 다른 시작점과 끝점을 가진 split_iter개의 sequantial 데이터가 만들어진다.
이후 모든 데이터를 전날 대비 등락률로 정규화한다.
```
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
      df_train2 = df_log2.iloc[:]
      df_test2 = df_log2.iloc[:]
```




* , table, graph, comparison, ...
* Web link

``` C++
void Example(int x, int y) {
   ...  
   ... // comment
   ...
}
```

## Conclusion
* Summary, contribution, ...

## Reports
* Upload or link (e.g. Google Drive files with share setting)
* Midterm: [Report](Reports/Midterm.pdf)
* Final: [Report](Reports/Final.pdf), [Demo video](Reports/Demo.mp4)
