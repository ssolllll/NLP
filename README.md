[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=ssolllll)](https://github.com/anuraghazra/github-readme-stats)

# NLP

- NLP 배운 내용을 토대로 실습
- 구글링을 통한 NLP 코드를 가지고 있는 데이터에 맞게 구현.
- 파인튜닝을 통한 성능 개선을 주안점으로 둠.
- Huggingface에 올라온 패키지 내용 정리

## NLP Task

1. PEFT(Parameter Fine-Tuning)
2. RAG(Retrieval Augmented Generation)
3. SFT(Supervised Fine-Tuning)
4. RLHF(Reinforcement Learning from Human Feedback)
5. NLP Task
    - Quantization(양자화)
    - Text Generation(텍스트 생성)
    - Language Translation(기계 번역)
    - Text Summarization(요약)
    - Question-Answering(질의응답)
    - Chatbot(대화 시스템)
    - Text Classification(문서 분류)
    - Sentiment Analysis(감성 분석)
    - Text Summarization(자동 요약)
    - Information Extraction(정보 추출)
    - Named Entity Recognition(개체명 인식)
6. etc
    - activation function
    - loss function
    - basic concept 

## Data

- 대학원 졸업논문에 사용한 IPO 관련 뉴스 기사 데이터
- 금융 뉴스 문장 감성 분석 데이터셋([출처](https://github.com/ukairia777/finance_sentiment_corpus))
- 크롤링을 통한 섹션별 뉴스 기사 데이터
- 네이버 x 창원대 NER 데이터([출처](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/naver_changwon_ner.html))
- Huggingface dataset


# 구조
📦NLP <br>
 ┣ 📂PEFT <br>
 ┃  ┣ 📜Concept.md <br>
 ┃  ┣ 📜PEFT_Bloom_tag.ipynb <br>
 ┃  ┣ 📜README.md <br>
 ┃  ┣ 📜T5_LoRA_tutorial.ipynb <br>
 ┣ 📂RAG <br>
 ┣ 📂RLHF <br>
 ┃  ┣ 📜README.md <br>
 ┣ 📂NLP_Task <br>
 ┃  ┣  📂Chatbot <br>
 ┃  ┣  ┗ 📜BART_chatbot_tuning_V3.ipynb <br>
 ┃  ┣  ┗ 📜Chatbot_tuning_V2.ipynb <br>
 ┃  ┣  ┗ 📜Chatbot_tuning.ipynb <br>
 ┃  ┣  📂NER <br>
 ┃  ┣  ┗ 📜NER_practice_v1.ipynb <br>
 ┃  ┣  ┗ 📜NER_practice_v2.ipynb <br>
 ┃  ┣  ┗ 📜NER_practice_v3.ipynb <br>
 ┃  ┣  📂Question_Answering <br>
 ┃  ┣  ┗ 📜korquad_preprocessing.py <br>
 ┃  ┣  ┗ 📜Question_Answering_v1.ipynb <br>
 ┃  ┣  ┗ 📜Question_Answering_v2_KorQuad.ipynb <br>
 ┃  ┣  📂Sentiment_Classification <br>
 ┃  ┣  ┗ 📜GPT2_Classification_by_tpu.ipynb <br>
 ┃  ┣  ┗ 📜Kobert_v1.ipynb <br>
 ┃  ┣  ┗ 📜Kobert_v2.ipynb <br>
 ┃  ┣  ┗ 📜transformer_sentiment_analysis.ipynb <br>
 ┃  ┣  📂Summarization <br>
 ┃  ┣  ┗ 📜Summarization_Data_preprocessing.ipynb <br>
 ┃  ┣  ┗ 📜summarization.py <br>
 ┃  ┣  ┗ 📜Text_summarization.ipynb <br>
 ┃  ┣ 📂Text Generation <br>
 ┃  ┣  ┗ 📜GPT2_Text_Generation.ipynb <br>
 ┃  ┣ 📂Translation <br>
 ┃  ┣  ┗ 📜Text_Translation_Data_preprocessing.ipynb <br>
 ┃  ┣  ┗ 📜Text_Translation_kor_en.ipynb <br>
 ┃  ┣  ┗ 📜translation_train.py <br>
 ┣ 📂etc  <br>
 ┃  ┣  ┗ 📜Basic Concepts for NLP.py <br>
 ┃  ┣  ┗ 📜Code Analysis for NLP.py <br>