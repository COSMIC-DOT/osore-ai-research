# osore-ai-research
# How to use Open-Source LLMs in Our project

colab url: https://colab.research.google.com/drive/1ysL8ArnFan7SoPShvXyRNT0Q_7lrGx9l?usp=sharing

### RESEARCH

[8 Top Open-Source LLMs for 2024 and Their Uses](https://www.datacamp.com/blog/top-open-source-llms)

Open-Source LLM중 가장 많이 쓰이고 추천 받는 모델은 LLaMA 2가 1등임

> it is a generative text model that can be used as a chatbot and can be adapted for a variety of natural language generation tasks, **including programming tasks**. Meta has already launched to open, customized versions of LLaMA 2, Llama Chat, and [**Code Llama.](https://ai.meta.com/blog/code-llama-large-language-model-coding/)
본문을 참고하자면 프로그래밍 업무또한 잘 한다고 한다.**
> 

### **LLaMA model - hugging face**

[NousResearch/Llama-2-7b-chat-hf · Hugging Face](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)

### **** How to Fine-tunning LLaMA

[Fine-Tuning LLaMA 2: A Step-by-Step Guide to Customizing the Large Language Model](https://www.datacamp.com/tutorial/fine-tuning-llama-2)

LLaMA 모델을 가져와서 fine-tunning하는 방식으로 진행할 수 있음

dataset은 hugging face에서 내가 원하는 dataset을 찾아서 fine tunning  시킬 수 있을 듯 

/

→ 찾아낸 dataset

# **GitHub Code Dataset**

[codeparrot/github-code · Datasets at Hugging Face](https://huggingface.co/datasets/codeparrot/github-code)

## **Dataset Description**

The GitHub Code dataset consists of 115M code files from GitHub in 32 programming languages with 60 extensions totaling in 1TB of data. The dataset was created from the public GitHub dataset on Google BiqQuery.

### **How to use it**

The GitHub Code dataset is a very large dataset so for most use cases it is recommended to make use of the streaming API of `datasets`. You can load and iterate through the dataset with the following two lines of code:

---

코드를 just 분석? 생성해주는 ai는?

### LLaMA- CodeLLaMA (코드 생성 AI)

[Introducing Code Llama, a state-of-the-art large language model for coding](https://ai.meta.com/blog/code-llama-large-language-model-coding/)

**codellama git**

[GitHub - meta-llama/codellama: Inference code for CodeLlama models](https://github.com/meta-llama/codellama)

**codellama paper**

[code_llama.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/445f7923-285e-4bba-803e-7cf813330a1d/code_llama.pdf)

**codellama model download**

[Download Llama](https://llama.meta.com/llama-downloads/)

### **LLAMA2 모델에서 CODE training 하는 architecture**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/a2cb7800-e52b-43dc-9993-9781520f1954/Untitled.png)

---

**라이선스 권리 및 재배포.
 a. 권리 부여. Meta의 지적 재산권 또는 Meta가 소유한 기타 권리에 기반한 Llama 자료에 대해 비독점적이며, 전 세계적이고, 양도 불가능하며, 로열티 없는 제한된 라이선스를 부여받습니다. 이를 통해 Llama 자료를 사용, 복제, 배포, 복사, 파생 작품을 만들고, Llama 자료를 수정할 수 있습니다.
 b. 재배포 및 사용.
     i. 제3자에게 Llama 자료 또는 그 파생 작품을 배포하거나 사용할 수 있게 할 경우, 해당 제3자에게 이 계약 사본을 제공해야 합니다.
     ii. 최종 사용자 제품의 일부로서 라이선스를 받은 자로부터 Llama 자료 또는 그 파생 작품을 받은 경우, 이 계약의 제2조는 귀하에게 적용되지 않습니다.
     iii. 배포하는 Llama 자료의 모든 복사본에는 다음과 같은 저작권 고지를 “Notice” 텍스트 파일 내에 포함하여 유지해야 합니다: “Llama 2는 LLAMA 2 커뮤니티 라이선스에 따라 라이선스가 부여되며, 저작권은 © Meta Platforms, Inc.에 있으며, 모든 권리가 보호됩니다.”
     iv. Llama 자료의 사용은 적용 가능한 법률 및 규정(무역 준수 법률 및 규정 포함)을 준수해야 하며, 본 계약에 참조로 포함된 Llama 자료에 대한 사용 정책([https://llama.meta.com/use-policy에서](https://llama.meta.com/use-policy%EC%97%90%EC%84%9C) 확인 가능)을 준수해야 합니다.
     v. Llama 자료 또는 Llama 자료의 결과물 혹은 출력물을 사용하여 Llama 2 또는 그 파생 작품을 제외한 다른 대규모 언어 모델을 개선하는 데 사용해서는 안 됩니다.**

**추가 상업 조건.
 Llama 2 버전 출시일에 라이선스를 받는 자나 그의 계열사가 제공하는 제품이나 서비스의 월간 활성 사용자 수가 이전 달력 월에 7억 명을 초과하는 경우, Meta로부터 라이선스를 요청해야 하며, Meta는 전적인 재량으로 귀하에게 그러한 라이선스를 부여할 수 있습니다. 또한 Meta가 명시적으로 그러한 권리를 부여하지 않는 한 본 계약 하에 어떠한 권리도 행사할 수 없습니다.

보증의 부인. 
 적용 가능한 법률에 의해 필요한 경우를 제외하고, Llama 자료 및 그로부터의 결과물과 출력물은 어떠한 종류의 명시적 또는 묵시적 보증 없이 "있는 그대로" 제공됩니다. 이에는 제한 없이 소유권, 비침해, 상품성 또는 특정 목적에의 적합성에 대한 보증이 포함됩니다. Llama 자료의 사용 또는 재배포의 적절성을 결정하는 것은 전적으로 귀하의 책임이며, Llama 자료 및 그로부터의 결과물과 출력물의 사용과 관련된 모든 위험을 가정합니다.

책임의 제한. 
 어떠한 책임 이론 하에서도, 계약, 불법 행위, 과실, 제품 책임 또는 그 밖의 경우에 있어, 본 계약으로 인해 발생하는 어떠한 경우에도 Meta 또는 그 계열사는 직간접적, 특별, 결과적, 우발적, 모범적 또는 징벌적 손해나 손실 이익에 대해 책임지지 않습니다. 이는 Meta 또는 그 계열사가 그러한 손해의 가능성을 사전에 통지받았더라도 마찬가지입니다.**

일단 download request 넣어놨음 이메일이 오게 되면 그 이후의 내용을 update하겠음

라마 다운로드 링크

[](https://download.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNTV5NzNvcHlla2M5ZnF3MWE0dWN6MnU0IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZG93bmxvYWQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMjE1NTMwMX19fV19&Signature=QKjAhCl9NW7wFf22r0J07odSt62RIhf3wWiMrTKCe-7P7pas6IhIUa2sq1FcWuyp~UZ0gvfVz~FJXm2DzjVOWbItKCr40mC5GaDbsv8yTN~E8iW3bDNeVmzt6QqTid7q6CAmA5xDmFpVMB-OBN8TngOBwTeOU~NmpZzd-jVvczvn81wFmRj17XsLp7w27loGjowoll2mJT6Tg-1o-7qxyTBJgf5cgREOtulcvYg~sQE~wE70qXLgAkb~V0kTX95ugNkzzRVK0V2WItg7LQNXnVzXV5V7RA~u3Iq9zIL8BlNpUHvWqmeoLDKK6y7WLhQtm6N80DagHOpIGxhuuqtglQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=934262115091287)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/a8e9a1bd-b216-40d2-8b4f-14334bed9211/Untitled.png)

# 추가 자료

**Fine-tunning 방식**

### **Difference between QLoRA and LoRA for Fine-Tuning LLMs.**

[Difference between QLoRA and LoRA for Fine-Tuning LLMs.](https://medium.com/@sujathamudadla1213/difference-between-qlora-and-lora-for-fine-tuning-llms-0ea35a195535)

### **GPT3.5 fine Fine-tunning**

[How to Fine Tune GPT 3.5: Unlocking AI's Full Potential](https://www.datacamp.com/tutorial/how-to-fine-tune-gpt3-5)

### Sourcegraph Github

[Sourcegraph](https://github.com/sourcegraph)

### SourceGraph

**How SourceGraph does works**

[Introduction to Sourcegraph](https://www.youtube.com/watch?v=D2x037j3BZ4&t=11s)

**Cody AI**

[Cody AI demo with Beyang Liu - Sourcegraph](https://www.youtube.com/watch?v=5L6Ys522snA)

### Copilot

How Copilot actually works

[Copilot for Microsoft 365   How it ACTUALLY Works!](https://www.youtube.com/watch?v=F5wfOhnj0IU&t=392s)

라마에 랭체인

[Getting Started with LangChain and Llama 2 in 15 Minutes | Beginner's Guide to LangChain](https://www.youtube.com/watch?v=7VAGe32YptI)

오픈소스 LLM 성능 리더보드

[Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

라마 + 랭체인 사용법

[랭체인(langchain) + 허깅페이스(HuggingFace) 모델 사용법 (2)](https://teddylee777.github.io/langchain/langchain-tutorial-02/)

LLM -code benchmarker

[GitHub - terryyz/llm-benchmark: A list of LLM benchmark frameworks.](https://github.com/terryyz/llm-benchmark)

[Big Code Models Leaderboard - a Hugging Face Space by bigcode](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)

big code model leaderboard 를 보면 나와있음

[Benchmarking LLMs: How to Evaluate Language Model Performance](https://luv-bansal.medium.com/benchmarking-llms-how-to-evaluate-language-model-performance-b5d061cc8679)

Instruction tunning

[[LLaMA 관련 논문 리뷰] 01-FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS (Instruction Tuning)](https://velog.io/@heomollang/LLaMA-논문-리뷰-1-LLaMA-Open-and-Efficient-Foundation-Language-Models)

RAG. 

[](https://aws.amazon.com/ko/what-is/retrieval-augmented-generation/)

아키텍쳐

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/65274095-9a0f-4767-bf0e-d8ac32e581f5/Untitled.jpeg)

[실전! RAG 고급 기법 - Retriever (1)](https://youtu.be/J2AsmUODBak?si=X6vsiJy7EAPcuPFb)

[Langchain 강의 (1/n) - Langchain이 뭘까?](https://youtu.be/WWRCLzXxUgs?si=KglJ0GC4PynpCDS7)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/de594b52-f50c-4809-8a51-e2991aaef486/Untitled.png)

Open Ko-LLM Leaderboard / huggingface : 한글 대규모 언어모델 성능 확인

대부분의 언어모델은 70billion개의 파라미터로 튜닝 — 개인 pc에서 돌아가기에 너무 크기에 양자화(Quanntization)을 통한 경량화 필요

PDF를 읽어와서 기반으로 답변해주는것도 가능

WebResearchRetriever을 활용하면 GitHub Repo를 참고하여 답변해주는것이 가능

여기서 핵심은? 웹사이트(github)를 참조하여 답변을 하는데 웹사이트 위에 올라와있는 단어들을 임베딩해서 토큰화 해서 원하는 질문에 대한 답변을 잘 만들어 낼 수 있을 것인가? → 모델 자체의 성능이 좋은 모델이라면? 당연히 가능 

RAG 방식으로 질문에 대한 답변의 참조를 만들어주는 것 까지는 okay 모델의 성능이 좋으려면 Parameter의 개수가 많은 Llama3와 같은 모델을 돌려야하는데 이 모델을 돌릴 GPU는 어디에? !! 

연구실에서 돌린다 or GPU서버를 구입해서 돌린다

RAG- WebResearchRetriever - 진행 과정 및 예시 사진 ppt에 포함되면 좋을 듯

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/0a65289b-db49-4fbc-82c5-9e4337166959/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/377216b6-1ab5-456c-9361-f7d8584f7fee/Untitled.png)

[LangChain | WebResearchRetriever을 활용하여 RAG (Retrieval Augmented Generation) 구현하기](https://littlefoxdiary.tistory.com/116)

RAG방식은 여러가지의 콘텐츠를 참조할 수 있는데 PDF, PPT, WORD, HWP, TXT, WEBSITE 등 많은 콘텐츠를 참조할 수 있는데 우리는 Website를 참조하는 방식을 채택 → 어떻게 자세히 code에 관한 내용을 임베딩할 수 있을까? → 이것과 상관없이 성능은 동일할 수도

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/8c1b22f4-4366-4c3c-8ddb-6db9d86ca5df/Untitled.png)

[GitHub - kyopark2014/rag-code-generation: It decribes code generation using RAG.](https://github.com/kyopark2014/rag-code-generation)

[Code understanding | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/use_cases/code_understanding/)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/3406f635-dbc3-4411-9525-0df4ad532420/Untitled.png)

Github에서 code를 clone해 와서 해당 레포를 .txt로 변환 이후에 chunk로 나누고 vector화 시킨다

[Chat with your code using RAG! - a Lightning Studio by akshay](https://lightning.ai/lightning-ai/studios/chat-with-your-code-using-rag?path=cloudspaces/01hqv3vhhramx0jb4bgq0gb8a0&tab=overview)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/20f5274d-4e20-45cb-a2bf-b8058e8670c1/Untitled.png)

현 시점에서 결과를 보려면 google colab에서 돌릴 수 있어야하는데 

PPT 들어갈 내용 정리

1. Github repository clone → 저장소에 저장
2. 저장된 repository에서 경로를 설정한 후 해당 경로 내부의 documents와 file명으로 ex) .py 지정한 코드를 가져와서 txt로 변환 → chunking
3. txt로 변환된 코드들을 openai embedding model에 넣고 vector화 시킴 →여기서 openai embedding model을 사용하면서 open api 소모 발생
4. vector화된( torch화) 데이터들을 langchain의 chroma를 사용해서 retriever를 만들어줌
5. 만들어낸 retriever를 가지고 chain을 하나 만들어줌 → {chat_history}를 통해 미리 질문했던 내용들을 기억하게 함
6. documents_chain을 구성 → context를 생성해내는 것
7. retriever chain과  documents chain을 연결해서 qa라는 변수에 저장 → context(chat history)를 반영해서 질문에 대한 답변을 작성해 낼 수 있도록 함
8. 모델을 준비하고 ( 모델은 현재 chat-gpt4를 활용, huggingface에서 가져온 모델이 invoke되지 않는 에러 발생으로 인해, invoke란 모델에게 질문하는것) 만들어놓은 langchain QA를 통해 모델에게 질문
9. 결과 확인

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/bf796975-5b7a-47e5-a485-8a0a0f2f5988/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/4e3e0ac2-ccb9-4db2-b942-ac07575827c3/751944dd-d86b-4620-876b-e6c0eabd0118/Untitled.png)

- > **Question**: Runnable 클래스에서 파생된 클래스는 무엇인가요?

**Answer**: 이 코드 스니펫에서 설명하는 Runnable 클래스에서 파생된 클래스를 명확하게 명시하지 않고 있습니다. 그러나 주석과 일부 클래스 정의에서 Runnable을 기반으로 하는 몇 가지 클래스를 추론할 수 있습니다. 예를 들어, RunnableLearnable, DynamicRunnable, BaseChatModel, BaseRetriever 등이 있습니다. 이러한 클래스들은 Runnable의 기능을 상속받고 확장하여 특정 작업을 수행합니다.

- > **Question**: Runnable 클래스의 클래스 계층 구조와 관련하여 어떤 개선을 제안하시겠습니까?

**Answer**: 코드 기반으로 제안할 수 있는 클래스 계층 구조에 대한 개선 사항은 다음과 같습니다:

1. Runnable 클래스에는 많은 메서드와 속성이 포함되어 있습니다. 이들을 더 작은, 특정 목적을 가진 클래스로 분할하여 코드의 가독성과 관리성을 향상시킬 수 있습니다.
2. 런타임에 특정 필드를 구성하는 기능을 제공하는 RunnableConfigurableFields 클래스를 만들어서 공통 기능을 재사용할 수 있게 할 수 있습니다. 이렇게 하면 코드 중복을 줄이고 유지 관리를 용이하게 할 수 있습니다.
3. Serializable, Runnable, RunnableGenerator와 같은 여러 클래스가 있습니다. 이들의 공통 요소를 추상 기본 클래스나 인터페이스로 분리하여 코드 재사용성을 향상시킬 수 있습니다.
4. Runnable 클래스가 현재 다양한 기능을 수행하고 있습니다. 이를 분리하여 각각의 클래스가 단일 책임 원칙을 따르도록 만들면 코드의 간결성과 가독성이 향상됩니다.
5. 현재 Runnable 클래스는 다양한 타입의 입력을 받을 수 있습니다. 이를 위해 많은 타입 검사와 분기 처리가 필요합니다. 이를 개선하기 위해, 입력을 받는 방식을 표준화하고, 이를 처리하는 별도의 클래스나 함수를 만들 수 있습니다.

이러한 제안사항들은 코드의 가독성, 재사용성, 유지보수성을 향상시키는 데 도움이 될 수 있습니다. 하지만 실제 프로젝트에서는 이러한 변경이 필요한지, 그리고 어떻게 구현할지에 대해 팀 내에서 논의가 필요합니다.

필요한 작업 → 모델은 code analysis에 특화된 모델을 만들어서 (hugginface에서 모델을 받아 해당 모델에 code에 특화된 학습을 추가로 진행) 서버에 올려놓고 돌아가게 만들고, github에서 clone해와서 해당 레포를 chunking하고 embedding하는 작업을 백엔드해서 수행한 후 langchain을 하는 서버로 전송해서 langchain해서 모델에 질문하고 답변받고 다시 프론트로 내보내주는 구성을 하면 끝.
