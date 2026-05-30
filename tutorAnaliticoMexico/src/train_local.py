import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer

# 1. Configuración estricta para GTX 1650 (4GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # float16 es más estable en la serie 16xx
)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print("Cargando Llama 3.2 (3B) en memoria...")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# 2. Configurar adaptadores LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. Cargar Dataset
dataset = load_dataset("json", data_files="../data/dataset_tutor.jsonl", split="train")

# 4. Entrenar (Protección contra OOM)
training_args = TrainingArguments(
    output_dir="./tutor_model",
    per_device_train_batch_size=1,       # CRÍTICO: Procesar de 1 en 1
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100,
    gradient_checkpointing=True,         # CRÍTICO: Ahorra VRAM sacrificando algo de velocidad
    optim="paged_adamw_8bit"             # CRÍTICO: Optimizador de bajo perfil de memoria
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="messages",
    max_seq_length=512,                  # CRÍTICO: Longitud máxima estricta
    args=training_args,
)

print("Iniciando Fine-Tuning local en GTX 1650...")
trainer.train()

# 5. Fusión Automática para exportación
print("Entrenamiento completado. Liberando memoria...")
trainer.model.save_pretrained("./tutor_adaptadores")

# Destruir objetos en VRAM para hacer espacio para la fusión
del model, trainer 
gc.collect()
torch.cuda.empty_cache()

print("Fusionando modelo base con adaptadores...")
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    "./tutor_adaptadores", torch_dtype=torch.float16, device_map="auto"
).merge_and_unload()

merged_model.save_pretrained("../data/tutor_unificado")
tokenizer.save_pretrained("../data/tutor_unificado")
print("¡Éxito! Modelo unificado guardado en data/tutor_unificado.")