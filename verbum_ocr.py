# Writing a script that uses LLMs to perform OCR on a PDF and shows the Markdown output!
# Code taken and adapted from https://github.com/yigitkonur/llm-ocr
# Will have to make the code modular, by that I mean, the LLM model can be easily swapped out

import asyncio
import argparse
import base64
import json
import logging
import os
from typing import Any, Callable, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pymupdf  # PyMuPDF, used to be called fitz
from dotenv import load_dotenv

# LLM Library Imports
import google.genai as genai
from google.genai import types
from openai import AsyncOpenAI

semaphore = asyncio.Semaphore(5)

parser = argparse.ArgumentParser()
parser.add_argument("--output_name", help="Name of the output files", type=str)
parser.add_argument("--model_name", help="Name of the model to be used", type=str, required=True)
parser.add_argument("pdf_input", help="PDF input to perform OCR on", type=str)
args = parser.parse_args()

# READ DOTENV FILE
load_dotenv(dotenv_path="config.env")

required_vars = ["PROMPT_DIR", "LLM_BASE_URL", "LLM_API_KEY"]

def check_required_vars(vars_list):
    for var in vars_list:
        if os.getenv(var) is None:
            raise ValueError(f"Required environment variable '{var}' is missing!")

# Check that all required variables are set
check_required_vars(required_vars)

def read_file(file_path):
    # Read markdown file, and return it as a str variable
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
        return data

PROMPT = read_file(os.getenv("PROMPT_DIR"))
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# LOGGING CONFIGURATION - CODE ADAPTED FROM Yigit Konur
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

def convert_page_to_image(args: Tuple[str, int, int]) -> Tuple[int, bytes]:
    """
    CODE ADAPTED FROM: Yigit Konur
    Convert a single PDF page to PNG image bytes using PyMuPDF.

    Args:
        args (Tuple[str, int, int]): A tuple containing:
            - pdf_path (str): Path to the PDF file.
            - page_num (int): Page number to convert (0-based).
            - zoom (int): Zoom factor for rendering.

    Returns:
        Tuple[int, bytes]: A tuple of page number and image bytes.

    Raises:
        Exception: If rendering fails.
    """
    pdf_path, page_num, zoom = args
    try:
        doc = pymupdf.open(pdf_path)
        page = doc.load_page(page_num)
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        image_bytes = pix.tobytes("png")
        logger.debug(
            f"Rendered page {page_num + 1}/{doc.page_count}, size: {len(image_bytes)} bytes."
        )
        doc.close()
        return (page_num + 1, image_bytes)  # Page numbers start at 1
    except Exception as e:
        logger.error(f"Error rendering page {page_num + 1}: {e}")
        raise

def convert_pdf_to_images_pymupdf(pdf_path: str, zoom: int = 2) -> List[Tuple[int, bytes]]:
    """
    CODE ADAPTED FROM: Yigit Konur
    Convert a PDF file to a list of PNG image bytes using PyMuPDF with multiprocessing.

    Args:
        pdf_path (str): Path to the PDF file.
        zoom (int): Zoom factor for rendering.

    Returns:
        List[Tuple[int, bytes]]: List of tuples containing page number and PNG image bytes.

    Raises:
        HTTPException: If conversion fails.
    """
    try:
        doc = pymupdf.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        logger.info(f"PDF loaded with {page_count} pages.")

        # Prepare arguments for each page
        args_list = [(pdf_path, i, zoom) for i in range(page_count)]

        image_bytes_list: List[Tuple[int, bytes]] = []  # List of (page_num, image_bytes)

        with ProcessPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(convert_page_to_image, args): args[1]
                for args in args_list
            }

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_num_result, image_bytes = future.result()
                    image_bytes_list.append((page_num_result, image_bytes))
                except Exception as e:
                    logger.error(f"Failed to convert page {page_num + 1}: {e}")
                    raise

        # Sort the list by page number to maintain order
        image_bytes_list.sort(key=lambda x: x[0])

        logger.info(f"Converted PDF to {len(image_bytes_list)} images using PyMuPDF.")
        return image_bytes_list

    except Exception as e:
        logger.exception(f"Error converting PDF to images with PyMuPDF: {e}")
        raise

def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    CODE ADAPTED FROM: Yigit Konur
    Encode image bytes to a base64 data URL.

    Args:
        image_bytes (bytes): The image content.

    Returns:
        str: Base64 encoded data URL.

    Raises:
        HTTPException: If encoding fails.
    """
    try:
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        #data_url = f"data:image/png;base64,{base64_str}"
        data_url = base64_str
        logger.debug(
            f"Encoded image to base64 data URL, length: {len(data_url)} characters."
        )
        return data_url
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise 

def encode_images(image_bytes_list: List[Tuple[int, bytes]]) -> List[Tuple[int, str]]:
    """
    Encode a list of image bytes to base64 data URLs along with their page numbers.

    Args:
        image_bytes_list (List[Tuple[int, bytes]]): List of tuples containing page numbers and image bytes.

    Returns:
        List[Tuple[int, str]]: List of tuples containing page numbers and base64-encoded image URLs.
    """
    encoded_urls = [(page_num, encode_image_to_base64(img_bytes)) for page_num, img_bytes in image_bytes_list]
    logger.info(f"Encoded {len(encoded_urls)} images to base64 data URLs.")
    return encoded_urls

async def call_gemini(args: Tuple[int, str], model: str, prompt: str) -> Tuple[int, str]:
    """
    Call Google Gemini
    Going to make it work for one page before doing it on multiple
    """
    # DO ERROR CHECKING TO ENSURE THAT MODEL IS IN MODELS!
    valid_models = ["gemini-pro-vision", "gemini-2.5-flash"]
    page_num, page_image_base64 = args
    
    image_part = types.Part.from_bytes(
        data=base64.b64decode(page_image_base64),
        mime_type="image/png"
    )
    
    # Asynchronous code adapted from:
    # https://github.com/google-gemini/cookbook/blob/main/quickstarts/Asynchronous_requests.ipynb
    try:
        client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY)
        response = await client.aio.models.generate_content(
            model=model,
            contents=[prompt, image_part]
        )
        return (page_num, response.text)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return "NULL"

async def call_vllm(args: Tuple[int, str], model: str, prompt: str) -> Tuple[int, str]:
    """
    Call model hosted on vLLM
    Using the OpenAI Python library to call locally hosted models
    """
    # DO ERROR CHECKING TO ENSURE THAT MODEL IS IN MODELS!
    page_num, page_image_base64 = args
    
    image_part = types.Part.from_bytes(
        data=base64.b64decode(page_image_base64),
        mime_type="image/png"
    )
    
    client = AsyncOpenAI(base_url=f"{os.getenv('LLM_BASE_URL')}", api_key=f"{LLM_API_KEY}")
    # Asynchronous code adapted from:
    # https://github.com/google-gemini/cookbook/blob/main/quickstarts/Asynchronous_requests.ipynb
    max_retries = 10
    retry_delay = 2 # Number of seconds before retrying
    for attempt in range(1, max_retries + 1):
        try:
            # Optional argument: api_key=f"{os.getenv("LLM_API_KEY")}
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{page_image_base64}"
                                }
                            }
                        ]
                    }
                ]
                # OPTIONS COMMENTED OUT FOR CHATGPT
                #temperature=0.2
                #max_tokens=4096
            )
            token_count = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            content = response.choices[0].message.content
            if not content:
                  raise ValueError(f"{page_num}: Response content was None or empty")
                
            return (page_num, f"{content}", token_count)
        except Exception as e:
            print(f"[Attempt {attempt}/{max_retries}] Error: {e}")

            if attempt < max_retries:
                wait_time = retry_delay * attempt
                print(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print("Max retries reached. Returning NULL.")
                return (page_num, "NULL", {})
        
        return (page_num, "NULL", {})

async def call_vllm_with_semaphore(page_data, model_name, prompt):
    async with semaphore:
        return await call_vllm(page_data, model_name, prompt)

async def main():
    pdf_pages = convert_pdf_to_images_pymupdf(args.pdf_input)
    encoded_pages = encode_images(pdf_pages)

    # TODO: Write code that is able to swap out the `call_gemini` function, for different models!
    # Might have to write an adapter of some kind!
    tasks = []
    for page_data in encoded_pages:
        task = asyncio.create_task(
            call_vllm_with_semaphore(page_data, args.model_name, PROMPT)
        )
        tasks.append(task)
        
    logger.info(f"\nDispatching {len(tasks)} API calls concurrently!")
    results = await asyncio.gather(*tasks)
    
    logger.info("All tasks are finished, exporting results")
    # Sort results by page number
    results.sort(key=lambda x: x[0])
    
    # Output the results as JSON and Markdown
    full_response_str = ""
    full_response_json = []
    for page_num, response_text, token_count in results:
        full_response_json.append({"page_num": page_num, "markdown": response_text, "token_count": token_count, "llm_model": args.model_name})
        full_response_str += response_text + "\n"

    #output_name = args.output_name
    date = datetime.now().strftime("%Y-%m-%d-T%H%M%S")
    output_name = f"{args.pdf_input.rstrip(".pdf")}_{date}"
    with open(f"test_{output_name}.json", 'w', encoding='utf-8') as f:
        json.dump(full_response_json, f, indent=4)
    
    with open(f"output_{output_name}.md", 'w', encoding='utf-8') as f:
        f.write(full_response_str)

if __name__ == "__main__":
    asyncio.run(main())