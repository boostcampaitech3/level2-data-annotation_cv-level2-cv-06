# 🍣 오마카세 
![image](https://user-images.githubusercontent.com/91659448/164880553-7433c1eb-b1e9-46b7-9abd-06cf63db554c.png)
- 대회 기간 : 2022.04.14 ~ 2022.04.21
- 목적 : 학습 데이터 추가 및 수정을 통한 이미지 속 글자 검출 성능 개선 대회

## 📝 글자 검출 대회
### 🔎 배경
- 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 
- 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

- OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다.


---

### 💾 데이터 셋
- [ICDAR2017_Mlt](https://rrc.cvc.uab.es/?ch=8) 데이터 셋 : 총 9000 장 중 한글과 영어로 이루어진 1063 장
- [ICDAR19_ArT](https://rrc.cvc.uab.es/?ch=14) 데이터 셋 : 총 5603 장 중 영어로 이루어진 2846 장
- [야외 실제 촬영 한글 이미지](https://aihub.or.kr/aidata/33985) 데이터 셋 샘플: 한글로 된 1140 장 전체

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

### 데이터 관점

| Data | LB Score@public | LB Score@private |
| :-: | :-: | :-: |
|ICDAR17 Korean|0.4882|0.4590|
|ICDAR17 MLT|0.6464|0.5980|
|ICDAR17 MLT, aihub 야외 간판 데이터|0.4672|0.5171|
|ICDAR17 MLT, ICDAR19 ArT|0.5502|0.5737|



- **결론** **:** ICDAR17 Korean만 사용했을 때보다 외부 데이터를 추가할수록 private score에서 좋은 성능을 보였다.
<br>

### Hyperparameter & Augmentation 관점

- ICDAR17 Korean 데이터
    - resize = 1200, crop = 256 
        - LB Score@public : 0.3800
        - LB Score@private : 0.4277
    - resize = 1200, crop = 800
        - LB Score@public :  0.3601  
        - LB Score@private : 0.3671   
<br>
- ICDAR17 Korean 데이터
    - Adam
        - LB Score@public : 0.4486
        - LB Score@private : 0.4686
    - AdamW
        - LB Score@public : 0.4723
        - LB Score@private : 0.4848 
<br>
- ICDAR17, ICDAR19 데이터
    - epoch=200
        - LB Score@public : 0.5940
        - LB Score@private : 0.5981
    - epoch=300
        - LB Score@public : 0.6141 
        - LB Score@private : 0.6295

---

## Reference
- [ICDAR2017_Mlt](https://rrc.cvc.uab.es/?ch=8)
- [ICDAR19_ArT](https://rrc.cvc.uab.es/?ch=14)
- [야외 실제 촬영 한글 이미지](https://aihub.or.kr/aidata/33985)


