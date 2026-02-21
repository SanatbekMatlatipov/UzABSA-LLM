# Data Directory

This directory contains raw and processed data for the UzABSA-LLM project.

## Structure

```
data/
├── raw/                    # Raw, unprocessed data
│   └── README.md
└── processed/              # Processed datasets ready for training
    └── README.md
```

## Data Sources

1. **Hugging Face Dataset** (Primary)
   - ID: `Sanatbek/aspect-based-sentiment-analysis-uzbek`
   - Size: 6,000 annotated reviews
   - Format: Structured ABSA annotations

2. **Raw Reviews** (Future semi-supervised learning)
   - Source: sharh.commeta.uz
   - Size: ~5,000 reviews
   - Format: Raw text without annotations

## Data Processing

Run the data preparation script to download and process the dataset:

```bash
python -m src.data_prep --output-dir ./data/processed
```

This will:
1. Download the dataset from Hugging Face
2. Format it for instruction tuning
3. Create train/validation splits
4. Save the processed data

## Data Format

The processed dataset contains a single `text` column with ChatML-formatted conversations:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

Where `output` is a JSON dictionary:
```json
{
    "aspects": [
        {
            "term": "aspect term",
            "category": "category",
            "polarity": "positive/negative/neutral"
        }
    ]
}
```
