# Verbum OCR
OCR tool that leverages the power of LLMs to intelligently convert scanned PDF documents to markdown format 

Code adapted from [yigitkonur/llm-ocr](https://github.com/yigitkonur/llm-ocr)

Utilises the OpenAPI Python library, to make requests to any LLM/VLM API utilising the OpenAPI specification.

Examples include:
- ChatGPT models via [OpenAPI Platform](https://platform.openai.com/docs/api-reference/responses/create)
- [Google Gemini](https://ai.google.dev/gemini-api/docs/openai)
- Models hosted with Ollama and vLLM. 

A Jupiter Notebook called `Verbum_OCR_Setup.ipynb` is included providing instructions on how to configure VerbumOCR to use various LLM models

`config.env` output:
```
PROMPT_DIR=[INSERT_PROMPT_DIR_HERE]
GOOGLE_GEMINI_API_KEY=[INSERT_GEMINI_API_KEY_HERE]
LLM_BASE_URL=[INSERT_BASE_URL_HERE]
LLM_API_KEY=[INSERT_API_KEY_HERE]
```
## Running the script
Example:
```bash
$ python verbum_ocr.py --model_name gpt5 sample.pdf 
```
