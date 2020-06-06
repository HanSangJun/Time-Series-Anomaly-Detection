# Time-series Anomaly Detection

정상 데이터로 모델링하여 비정상을 검출하는 One-class 기반 Time-series Anomaly Detection 알고리즘에 대해 알아보겠습니다. Anomaly Detection은 여러 공정에서 예지보전에 기본이 되는 알고리즘이며, 시간적 패턴과 센서 간의 관계성을 잘 표현하는 것이 중요합니다. 여기서 소개되는 알고리즘은 다음과 같은 과정을 거쳐 비정상을 검출합니다.

1. 정상 데이터로 모델링 수행
2. 비정상을 판단하는 통계적 threshold 값 설정
3. 설정한 threshold 값으로부터 비정상 추정

<strong>참조:</strong> <em>S. Yin et al., A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman Process, J. Process Control, 2012</em>

## Dataset

본 예제에서 사용하는 dataset은 화학 공정을 모사하여 시뮬레이션 생성한 Tennesse Eastman Process 2001입니다. Train set과 test set으로 나누어져 있으며, 정상 상태 포함 총 22가지의 fault type을 제공하고 있습니다. 온도, 압력, 유량 등의 52개의 센서로 구성되어 있고, 3분마다 sampling 됩니다. Train set은 24시간, test set은 48시간으로 구성됩니다. (자세한 사항은 논문 참조)


## Algorithms

* PCA (pca_main.ipynb)
* DPCA (TBA)
* CVA (TBA)
* ICA (ica_main.ipynb)
* AutoEncoder (TBA)
* LSTM-AutoEncoder (TBA)


## Authors

* 한상준 (hjun1008@gmail.com)
* Last update: 2020-03-30
