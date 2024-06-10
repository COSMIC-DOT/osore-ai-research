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

# 추가
