# 🍣 오마카세 
![image](https://user-images.githubusercontent.com/91659448/164880553-7433c1eb-b1e9-46b7-9abd-06cf63db554c.png)
- 대회 기간 : 2022.04.14 ~ 2022.04.21
- 목적 : 학습 데이터 추가 및 수정을 통한 이미지 속 글자 검출 성능 개선 대회

## 📝 글자 검출 대회
### 🔎 배경
- 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식 하는 등 OCR은 실생활에서 널리 쓰이는 대표적인 기술입니다.
- 해당 기술을 Model-Centric이 아닌 Data-Centric을 통해 모델 성능을 올리는 방법을 알아보고자 합니다.


---

### 💾 데이터 셋
- `ICDAR2017_Mlt 데이터 셋` : 총 9000 장 중 한글과 영어로 이루어진 1063 장
- `ICDAR19_ArT 데이터 셋` : 총 5603 장 중 영어로 이루어진 2846 장
- `야외 실제 촬영 한글 이미지 데이터 셋`: 샘플 데이터 1140 장

---

## 🙂 멤버
| 박동훈 | 박성호 | 송민기 | 이무현 | 이주환 |  
| :-: | :-: | :-: | :-: | :-: |  
|[Github](https://github.com/BTOCC25) | [Github](https://github.com/pyhonic) | [Github](https://github.com/alsrl8) | [Github](https://github.com/PeterLEEEEEE) | [Github](https://github.com/JHwan96)

---

## 📋 역할
| 멤버 | 역할 |
| :-: | :-: |
|박동훈(T3086)| ICDAR2017-MLT 데이터 셋 적용, EDA|
|박성호(T3090)| ICDAR2017-MLT + ICDAR19_ArT 데이터 셋 적용, EDA, utils 파일 생성|
|송민기(T3112)| ICDAR2017-MLT 데이터 셋 적용|
|이무현(T3144)| ICDAR2017-MLT + 야외 실제 촬영 한글 이미지 데이터 셋 적용|
|이주환(T3241)| ICDAR2017-MLT + 야외 실제 촬영 한글 이미지 데이터 셋 적용|

---

## 🧪 실험

### 📝 데이터 관점

| Data | LB Score@public | LB Score@private |
| :-: | :-: | :-: |
|ICDAR17 Korean|0.4882|0.4590|
|ICDAR17 MLT|0.6464|0.5980|
|ICDAR17 MLT, aihub 야외 간판 데이터|0.4672|0.5171|
|ICDAR17 MLT, ICDAR19 ArT|0.5502|0.5737|



- ICDAR17 Korean 만 사용했을 때보다 외부 데이터를 추가할수록 private score에서 좋은 성능을 보였습니다.
<br>

### 📌 Hyperparameter & Augmentation 관점

- ICDAR17 Korean 데이터 (Resize, Crop)
    - 1200, 256 (public: 0.3800, private: 0.4277)
    - 1200, 800 (public: 0.3601, private: 0.3671)

- ICDAR17 Korean 데이터 (Adam, AdamW)
    - Adam (public: 0.4486, private: 0.4686)
    - AdamW (public: 0.4723, private: 0.4848)
 
- ICDAR17, ICDAR19 데이터 (epoch 200, epoch 300)
    - 200 (public: 0.5940, private: 0.5981)
    - 300 (public: 0.6141, private: 0.6295)


---

## Reference
- [ICDAR2017_Mlt](https://rrc.cvc.uab.es/?ch=8)
- [ICDAR19_ArT](https://rrc.cvc.uab.es/?ch=14)
- [야외 실제 촬영 한글 이미지](https://aihub.or.kr/aidata/33985)


