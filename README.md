# Verbum OCR
OCR tool that leverages the power of LLMs to intelligently convert scanned PDF documents to markdown format 

Code adapted from [yigitkonur/llm-ocr](https://github.com/yigitkonur/llm-ocr)

So far, Verbum is programmed to only use Google Gemini. I plan on making the tool more modular, so that other LLM models can be swapped out in place of Gemini.

`config.env` output:
```
PROMPT_DIR=[INSERT_PROMPT_DIR_HERE]
GOOGLE_GEMINI_API_KEY=[INSERT_GEMINI_API_KEY_HERE]
LLM_BASE_URL=[INSERT_BASE_URL_HERE]
LLM_API_KEY=[INSERT_API_KEY_HERE]
```
