


# 📚 **Google Gemini Integration with LangChain: A Step-by-Step Guide**

## 💡 **Step 1: Install Required Libraries**
To start, install the necessary libraries. We'll be using `langchain` for the pipeline and `google-generativeai` to access the Google Gemini model.

```bash
!pip install langchain -q
!pip install google-generativeai -q
```
---

## 💡 **Step 2: Import Necessary Modules**
Next, import the required modules for creating the language model chain. 

```bash
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from langchain.llms.base import LLM

```

## 💡 **Step 3: Fetch and Configure API Key**
In this step, fetch the API key from the user data to authenticate with the Google Gemini API.

```bash
from google.colab import userdata

# Fetching API key directly from Colab userdata
GEMINI_API_KEY = userdata.get('GOOGLE_API_KEY')

# Check if the API key is present
if not GEMINI_API_KEY:
    raise ValueError("API key not found. Make sure 'GOOGLE_API_KEY' is set in user data.")

```

## 💡 **Step 4: Configure the API with the API Key**
Here, we configure the API with the key we retrieved in the previous step.

```bash
# Configure the API with the provided key
genai.configure(api_key=GEMINI_API_KEY)


```

## 💡 **Step 5: Create a Custom Gemini LLM Class**
Define a custom class GeminiLLM that inherits from LLM and specifies the behavior for interacting with the Gemini model.

```bash
class GeminiLLM(LLM):
    model = "gemini-pro"
    temperature = 0.7
    @property
    def _llm_type(self) -> str:
        return "google_gemini"
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return genai.GenerativeModel(self.model).generate_content
        (prompt, generation_config={"temperature": self.temperature}).text



```

## 💡 **Step 6: Initialize the LLM**
Here we initialize an instance of the GeminiLLM class, using the default model and temperature.

```bash
llm = GeminiLLM()


```

## 💡 **Step 7: Define a Prompt Template**
We now define a prompt template that will be used to structure the input for the language model.

```bash
prompt_template = PromptTemplate(
    input_variables=["question"],  # Variable to take user input
    template="You are a helpful assistant. Answer the following question:\n\n{question}"
)


```

## 💡 **Step 8: Create the LangChain Pipeline**
We now create a pipeline that connects the GeminiLLM with the defined PromptTemplate using LLMChain.

```bash
chain = LLMChain(llm=llm, prompt=prompt_template)


```

## 💡 **Step 9: Ask a Sample Question**
Finally, we ask a sample question and get the response from the model using the chain.

```bash
question = "What is LangChain?"
response = chain.run({"question": question})
print("Answer:", response)


```
## 🚀 **Result Below**

### **Answer**:
LangChain is a web-based platform that facilitates language learning through collaborative translation. It connects learners of different languages who help each other translate texts, providing feedback and corrections along the way. This interactive approach aims to enhance language proficiency, cultural understanding, and global collaboration.
