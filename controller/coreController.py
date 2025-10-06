from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import transformers

transformers.logging.set_verbosity_error()


class CoreController:
    def __init__(self, model_type="phi1.5", model_name=None):
        self.available_models = {
            "gpt2": "gpt2",
            "phi1.5": "microsoft/phi-1.5",
            "nous": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
            "qwen2.5": "Qwen/Qwen2-0.5B"
        }

        self.models = {}        # Cache of loaded models
        self.tokenizers = {}    # Cache of loaded tokenizers
        self.devices = {}       # Store devices per model

        self.sentiment_analysis = pipeline("sentiment-analysis")

        # Load default model
        self.current_model_key = model_type
        self._load_model(model_type, model_name)

    def _load_model(self, model_type, custom_model_name=None):
        """
        Load and cache a model/tokenizer. Use cached version if already loaded.
        """
        model_key = model_type.lower()
        model_name = custom_model_name or self.available_models.get(model_key)

        if not model_name:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_key in self.models:
            print(f"✅ Using cached model: {model_key}")
        else:
            print(f"🔄 Loading new model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                device = torch.device("cuda")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                device = torch.device("cpu")
                model.to(device)

            model.eval()

            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
            self.devices[model_key] = device

        # Set active model/tokenizer/device
        self.current_model_key = model_key
        self.model = self.models[model_key]
        self.tokenizer = self.tokenizers[model_key]
        self.device = self.devices[model_key]

    def switch_model(self, model_type, model_name=None):
        """
        Public method to switch models dynamically.
        """
        self._load_model(model_type, model_name)

    def conv(self, user_input, max_new_tokens=150):
        sentiment = self.sentiment_analysis(user_input)[0]['label']

        system_prompt = (
        )

        # Check if the tokenizer supports chat template
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"

        print(f"[{self.current_model_key}] Prompt:\n", prompt)

        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.1,
                )

            response = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
            print("Generated response:", response)
            return response.strip()

        except Exception as e:
            print(f"Error during generation with model '{self.current_model_key}': {str(e)}")
            return "I'm sorry, I couldn't generate a response. Please try again later."

