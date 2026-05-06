import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor, 
    MllamaForConditionalGeneration, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pathlib import Path

# --- DYNAMIC PATH CONFIGURATION ---
# Base directory for relative path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MODEL_PATH = SCRIPT_DIR / "models" / "base_model"
TRAIN_DATASET_PATH = SCRIPT_DIR / "data" / "screw_dataset_train.jsonl"
OUTPUT_ADAPTER_DIR = SCRIPT_DIR / "outputs" / "lora_weights"

def main():
    print("🚀 Initializing Multimodal QLoRA Fine-Tuning for Llama 3.2 Vision...")
    
    # 1. Quantization Configuration (4-bit) optimized for V100/A100 GPUs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # Critical for Volta architecture stability
    )

    print("Loading processor and model into memory...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    model = MllamaForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Performance optimization: Disable internal cache to prevent memory overhead
    model.config.use_cache = False 
    model = prepare_model_for_kbit_training(model)

    # 2. LoRA Configuration
    # Targeting specific projection layers for deep vision-language alignment
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Dataset Loading
    print(f"Loading training dataset: {TRAIN_DATASET_PATH}")
    dataset = load_dataset("json", data_files=str(TRAIN_DATASET_PATH), split="train")

    # 4. Multimodal Data Collator
    def multimodal_collate_fn(examples):
        batch_texts =
        batch_images =
        
        for item in examples:
            # Apply chat template for instruction-following model structure
            text_query = processor.apply_chat_template(item["messages"], tokenize=False)
            batch_texts.append(text_query)
            
            image_path = None
            for msg in item["messages"]:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "image":
                            image_path = content["image"]
                            break
            
            if image_path and os.path.exists(image_path):
                batch_images.append(Image.open(image_path).convert("RGB"))
            else:
                print(f"⚠️ Warning: Image file not found at {image_path}")
        
        inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)
        
        # Masking padding tokens in labels to avoid calculating loss on them
        labels = inputs["input_ids"].clone()
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
            
        # Specific token ID masking for Mllama architecture
        labels[labels == 128256] = -100
        inputs["labels"] = labels
        
        return inputs

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_ADAPTER_DIR),
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=8,  
        learning_rate=5e-5,             
        logging_steps=5,                
        max_steps=400,                  
        save_strategy="steps",
        save_steps=100,
        optim="paged_adamw_8bit",       
        fp16=True,                      
        remove_unused_columns=False,    
        report_to="none",               
        # Memory saving patch to avoid graph errors during backward pass
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=multimodal_collate_fn, 
    )

    print("\n🔥 Starting Multimodal Fine-Tuning...")
    trainer.train()

    # 7. Model Persistence
    print("\n💾 Saving learned LoRA adapters...")
    final_output = OUTPUT_ADAPTER_DIR / "final"
    trainer.model.save_pretrained(str(final_output))
    processor.save_pretrained(str(final_output))
    
    print(f"✅ Training completed. Artifacts saved in: {final_output}")

if __name__ == "__main__":
    main()
