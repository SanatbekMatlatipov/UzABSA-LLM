#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Inference Module
# =============================================================================
"""
Inference module for Uzbek Aspect-Based Sentiment Analysis.

This module provides utilities for:
1. Loading fine-tuned models (LoRA adapters or merged models)
2. Running inference on single texts or batches
3. Parsing model outputs into structured data

Author: UzABSA Team
License: MIT
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

import torch

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# System Prompts (must match training prompts)
# =============================================================================

SYSTEM_PROMPT_UZ = """Siz o'zbek tilida matnlardan aspektlarni va ularning hissiyotlarini aniqlash bo'yicha mutaxassissiz.

Berilgan matndan barcha aspekt terminlarini, ularning kategoriyalarini va hissiyot polaritesini (positive, negative, neutral) ajratib oling.

Javobni quyidagi Python dictionary formatida qaytaring:
{
    "aspects": [
        {
            "term": "aspekt termin",
            "category": "kategoriya",
            "polarity": "positive/negative/neutral"
        }
    ]
}"""

SYSTEM_PROMPT_EN = """You are an expert in extracting aspects and their sentiments from Uzbek text.

From the given text, extract all aspect terms, their categories, and sentiment polarity (positive, negative, neutral).

Return the response in the following Python dictionary format:
{
    "aspects": [
        {
            "term": "aspect term",
            "category": "category",
            "polarity": "positive/negative/neutral"
        }
    ]
}"""


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    device_map: str = "auto",
):
    """
    Load a fine-tuned model for inference.

    Supports loading:
    - Merged models (full model weights)
    - Base models with LoRA adapters
    - Standard HuggingFace models

    Args:
        model_path: Path to the model or HuggingFace model ID.
        adapter_path: Path to LoRA adapters (if separate from model).
        max_seq_length: Maximum sequence length.
        load_in_4bit: Whether to load in 4-bit quantization.
        device_map: Device placement strategy ("auto", "cuda", "cpu").

    Returns:
        Tuple of (model, tokenizer) ready for inference.

    Example:
        >>> model, tokenizer = load_model("./outputs/my_run/merged_model")
        >>> # Or with separate adapters:
        >>> model, tokenizer = load_model(
        ...     "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        ...     adapter_path="./outputs/my_run/lora_adapters"
        ... )
    """
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # Try loading with Unsloth (faster)
        from unsloth import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        
        # Load adapters if provided
        if adapter_path:
            logger.info(f"Loading LoRA adapters from: {adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
    except ImportError:
        logger.warning("Unsloth not available, using standard HuggingFace loading")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16 if load_in_4bit else torch.float32,
        )
        
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        
        model.eval()
    
    logger.info("Model loaded successfully!")
    return model, tokenizer


# =============================================================================
# Inference Functions
# =============================================================================

def create_inference_prompt(
    text: str,
    use_uzbek: bool = True,
) -> str:
    """
    Create a prompt for aspect extraction inference.

    Args:
        text: The input text to analyze.
        use_uzbek: Whether to use Uzbek prompts.

    Returns:
        Formatted ChatML prompt string.
    """
    system_prompt = SYSTEM_PROMPT_UZ if use_uzbek else SYSTEM_PROMPT_EN
    
    if use_uzbek:
        user_message = (
            f"Quyidagi o'zbek tilidagi matndan aspektlarni, "
            f"kategoriyalarni va hissiyot polaritesini aniqlang:\n\n"
            f"Matn: \"{text}\""
        )
    else:
        user_message = (
            f"Extract aspects, categories, and sentiment polarities "
            f"from the following Uzbek text:\n\n"
            f"Text: \"{text}\""
        )
    
    # Format as ChatML
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return prompt


def extract_aspects(
    model,
    tokenizer,
    text: str,
    use_uzbek: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """
    Extract aspects from a single text using the fine-tuned model.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        text: Input text to analyze.
        use_uzbek: Whether to use Uzbek prompts.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
        top_p: Nucleus sampling parameter.
        do_sample: Whether to use sampling (False = greedy decoding).

    Returns:
        Dictionary with extracted aspects.

    Example:
        >>> result = extract_aspects(model, tokenizer, "Bu telefon juda yaxshi!")
        >>> print(result)
        {"aspects": [{"term": "telefon", "category": "general", "polarity": "positive"}]}
    """
    # Create prompt
    prompt = create_inference_prompt(text, use_uzbek)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length - max_new_tokens,
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False),
        )
    
    # Decode
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    # Parse output
    result = parse_model_output(generated_text)
    
    return result


def extract_aspects_batch(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 4,
    use_uzbek: bool = True,
    max_new_tokens: int = 512,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract aspects from multiple texts in batches.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        texts: List of input texts.
        batch_size: Number of texts to process at once.
        use_uzbek: Whether to use Uzbek prompts.
        max_new_tokens: Maximum tokens to generate per text.
        show_progress: Whether to show progress bar.

    Returns:
        List of result dictionaries.
    """
    from tqdm import tqdm
    
    results = []
    
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing texts")
    
    for i in iterator:
        batch_texts = texts[i:i + batch_size]
        
        for text in batch_texts:
            result = extract_aspects(
                model=model,
                tokenizer=tokenizer,
                text=text,
                use_uzbek=use_uzbek,
                max_new_tokens=max_new_tokens,
            )
            results.append(result)
    
    return results


# =============================================================================
# Output Parsing Functions
# =============================================================================

def parse_model_output(output_text: str) -> Dict[str, Any]:
    """
    Parse the model's generated output into a structured dictionary.

    Handles various output formats and edge cases.

    Args:
        output_text: Raw text generated by the model.

    Returns:
        Parsed dictionary with aspects, or error information.

    Example:
        >>> output = '{"aspects": [{"term": "narx", "polarity": "negative"}]}'
        >>> result = parse_model_output(output)
        >>> print(result)
        {"aspects": [{"term": "narx", "polarity": "negative"}], "raw_output": "..."}
    """
    # Clean up the output
    cleaned = output_text.strip()
    
    # Remove common artifacts
    cleaned = cleaned.replace("<|im_end|>", "").strip()
    
    # Try to find JSON in the output
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    
    if json_match:
        try:
            result = json.loads(json_match.group())
            result["raw_output"] = output_text
            result["parse_success"] = True
            return result
        except json.JSONDecodeError:
            pass
    
    # Try to extract aspects even from malformed output
    aspects = extract_aspects_from_text(cleaned)
    
    return {
        "aspects": aspects,
        "raw_output": output_text,
        "parse_success": len(aspects) > 0,
    }


def extract_aspects_from_text(text: str) -> List[Dict[str, str]]:
    """
    Attempt to extract aspects from potentially malformed text.

    This is a fallback parser for when JSON parsing fails.

    Args:
        text: Text that may contain aspect information.

    Returns:
        List of aspect dictionaries (may be empty).
    """
    aspects = []
    
    # Pattern to match aspect entries
    patterns = [
        # Pattern for dict-like entries
        r'"term":\s*"([^"]+)".*?"category":\s*"([^"]+)".*?"polarity":\s*"([^"]+)"',
        # Pattern for simpler format
        r'term["\']?\s*[:=]\s*["\']?([^"\']+)["\']?.*?polarity["\']?\s*[:=]\s*["\']?(positive|negative|neutral)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            if len(match) >= 2:
                aspect = {
                    "term": match[0].strip(),
                    "polarity": match[-1].strip().lower() if len(match) > 1 else "neutral",
                }
                if len(match) >= 3:
                    aspect["category"] = match[1].strip()
                aspects.append(aspect)
    
    return aspects


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_text(
    text: str,
    model_path: str = None,
    model=None,
    tokenizer=None,
    use_uzbek: bool = True,
) -> Dict[str, Any]:
    """
    High-level function to analyze a single text.

    Can either load a model or use provided model/tokenizer.

    Args:
        text: Text to analyze.
        model_path: Path to model (if model/tokenizer not provided).
        model: Pre-loaded model (optional).
        tokenizer: Pre-loaded tokenizer (optional).
        use_uzbek: Whether to use Uzbek prompts.

    Returns:
        Analysis results dictionary.
    """
    if model is None or tokenizer is None:
        if model_path is None:
            raise ValueError("Either model_path or model/tokenizer must be provided")
        model, tokenizer = load_model(model_path)
    
    return extract_aspects(model, tokenizer, text, use_uzbek)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Interactive inference CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ABSA inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File with texts (one per line)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--uzbek", action="store_true", default=True, help="Use Uzbek prompts")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    if args.text:
        # Single text
        result = extract_aspects(model, tokenizer, args.text, args.uzbek)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.file:
        # File with multiple texts
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = extract_aspects_batch(model, tokenizer, texts, use_uzbek=args.uzbek)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
    
    else:
        # Interactive mode
        print("Interactive ABSA Inference")
        print("Enter text to analyze (Ctrl+C to exit):")
        print("-" * 50)
        
        while True:
            try:
                text = input("\nText: ").strip()
                if text:
                    result = extract_aspects(model, tokenizer, text, args.uzbek)
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            except KeyboardInterrupt:
                print("\nExiting...")
                break


if __name__ == "__main__":
    main()
