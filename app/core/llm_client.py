
from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import settings


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        ...


class GroqLLM(LLMClient):
    def __init__(self):
        from groq import Groq  # local import so this stays optional at import-time

        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and put it in your .env file."
            )
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


class LocalHFLLM(LLMClient):
  
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.local_model_name, token=settings.hf_token or None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.local_model_name,
            quantization_config=quant_config,
            device_map="auto",
            token=settings.hf_token or None,
        )

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        output = self.model.generate(
            input_ids,
            max_new_tokens=800,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
        )
        generated = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    if settings.llm_provider == "local":
        return LocalHFLLM()
    return GroqLLM()
