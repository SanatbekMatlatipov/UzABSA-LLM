# Optional: BDCC-oriented abstract opening (leads with the pipeline / data story)

Drop-in alternative for the first two sentences of the abstract, reframing the paper around
the "big data / cognitive computing" angle (scalable resource creation) rather than the model
comparison. Use only if you want to lean into venue fit; the current opening is acceptable.

> Building annotated resources for low-resource languages at scale is a central bottleneck for
> applied NLP. We address this for Uzbek aspect-based sentiment analysis (ABSA) with a scalable,
> reproducible pipeline that couples parameter-efficient fine-tuning of open-source large
> language models with automated quality control, and we validate its output against
> native-speaker annotation. As the pipeline's engine, we conduct — to our knowledge — the first
> systematic comparison of QLoRA-fine-tuned 7–8B LLMs (Qwen 2.5, Llama 3.1, DeepSeek-R1-Distill)
> for Uzbek ABSA, benchmarked against fine-tuned Uzbek BERT encoders …

Then continue with the existing sentences (dataset sizes, results judge scores). Keep total
abstract under ~250 words.

Note: MDPI abstracts must be a single unstructured paragraph with no citations and no line breaks — keep it one block when pasting into `main.tex`.
