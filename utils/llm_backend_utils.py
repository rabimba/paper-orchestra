# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from typing import Optional, Callable, Dict, Any

from google.genai import types

from utils.gemini_utils import (
    call_gemini_with_text_prompt,
    parse_gemini_json_results,
    call_gemini_with_contents,
    call_gemini_with_images,
)
from utils.openai_utils import (
    call_openai_models_with_text_prompt,
    call_openai_models_with_content,
    parse_openai_json_results,
    get_openai_image_part_from_path,
)
from utils.pdf_utils import load_paper


def is_openai_backend(model_name: str) -> bool:
    """Determines whether a model should be queried via the OpenAI-compatible client.
    
    Routes to OpenAI backend if:
    1. Model name includes standard OpenAI identifiers ('gpt', 'o1', 'o3', 'openai/').
    2. Environment explicitly sets LLM_BACKEND=openai.
    3. OPENAI_BASE_URL is set (e.g. puffingbird vLLM/Ollama server) and model is not Google Gemini.
    """
    name = model_name.lower()
    if (
        "gpt" in name
        or "o1" in name
        or "o3" in name
        or name.startswith("openai/")
        or os.environ.get("LLM_BACKEND", "").lower() == "openai"
    ):
        return True
    if os.environ.get("OPENAI_BASE_URL"):
        if not ("gemini" in name or name.startswith("google/")):
            return True
    return False


def identity_parse(response: str):
    return response


def get_llm_parser(model_name: str, return_json: bool = True) -> Callable:
    if not return_json:
        return identity_parse

    if is_openai_backend(model_name):
        return parse_openai_json_results
    else:
        return parse_gemini_json_results


def call_llm_with_text_prompt(
    prompt: str,
    model_name: str,
    generation_configs: Optional[Dict[str, Any]] = None,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Unified function to call either Gemini or OpenAI models based on the model_name.
    """
    parser = (
        result_parsing_func
        if result_parsing_func is not None
        else get_llm_parser(model_name, return_json)
    )

    if generation_configs is None:
        generation_configs = {}

    if is_openai_backend(model_name):
        system_prompt = None
        configs_for_openai = dict(generation_configs) if generation_configs else {}
        if "system_instruction" in configs_for_openai:
            system_prompt = configs_for_openai.pop("system_instruction")
        if "temperature" in configs_for_openai:
            configs_for_openai.pop("temperature")

        response_dict = call_openai_models_with_text_prompt(
            prompt=prompt,
            model_name=model_name,
            generation_configs=configs_for_openai,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
            system_prompt=system_prompt,
        )
        return response_dict
    else:
        response_dict = call_gemini_with_text_prompt(
            prompt=prompt,
            model_name=model_name,
            generation_configs=generation_configs,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
        )
        return response_dict


def call_llm_with_images(
    prompt: str,
    images: list[str],
    model_name: str,
    generation_configs: Optional[Dict[str, Any]] = None,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Unified function to call either Gemini or OpenAI models with images based on the model_name.
    """
    parser = (
        result_parsing_func
        if result_parsing_func is not None
        else get_llm_parser(model_name, return_json)
    )

    if generation_configs is None:
        generation_configs = {}

    if is_openai_backend(model_name):
        system_prompt = None
        # Copy to avoid side-effects on caller's dictionary
        configs_for_openai = dict(generation_configs) if generation_configs else {}
        if "system_instruction" in configs_for_openai:
            system_prompt = configs_for_openai.pop("system_instruction")
        if "temperature" in configs_for_openai:
            configs_for_openai.pop("temperature")

        content = [{"type": "text", "text": prompt}]
        for image_path in images:
            content.extend(get_openai_image_part_from_path(image_path))

        response_dict = call_openai_models_with_content(
            content=content,
            model_name=model_name,
            generation_configs=configs_for_openai,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
            system_prompt=system_prompt,
        )
        return response_dict
    else:
        response_dict = call_gemini_with_images(
            prompt=prompt,
            images=images,
            model_name=model_name,
            generation_configs=generation_configs,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
        )
        return response_dict


def call_llm_with_pdf(
    pdf_path: str,
    prompt: str,
    model_name: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Unified function to evaluate a PDF.
    Passes raw PDF bytes to Gemini, and extracted text to OpenAI.
    """
    parser = (
        result_parsing_func
        if result_parsing_func is not None
        else get_llm_parser(model_name, return_json)
    )

    if is_openai_backend(model_name):
        paper_text = load_paper(pdf_path, min_size=100)
        full_prompt = f"Paper Content:\n{paper_text}\n\nTask:\n{prompt}"

        generation_configs = {}
        response_dict = call_openai_models_with_text_prompt(
            prompt=full_prompt,
            model_name=model_name,
            generation_configs=generation_configs,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
            system_prompt=system_instruction,
        )
        return response_dict
    else:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        contents = [
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=prompt),
        ]

        generation_configs = {"temperature": temperature}
        if system_instruction is not None:
            generation_configs["system_instruction"] = system_instruction

        response_dict = call_gemini_with_contents(
            contents=contents,
            model_name=model_name,
            generation_configs=generation_configs,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
        )
        return response_dict


def call_llm_with_contents(
    contents: list,
    model_name: str,
    generation_configs: Optional[Dict[str, Any]] = None,
    result_parsing_func: Optional[Callable] = None,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
) -> Dict[str, Any]:
    """
    Unified function to call either Gemini or OpenAI/Ollama with a list of text parts / contents.
    """
    parser = (
        result_parsing_func
        if result_parsing_func is not None
        else get_llm_parser(model_name, return_json)
    )

    if is_openai_backend(model_name):
        text_parts = []
        for part in contents:
            if hasattr(part, "text") and part.text is not None:
                text_parts.append(part.text)
            elif hasattr(part, "inline_data") and part.inline_data is not None:
                if getattr(part.inline_data, "mime_type", "") == "application/pdf":
                    import pymupdf
                    doc = pymupdf.open(stream=part.inline_data.data, filetype="pdf")
                    extracted = "\n".join([page.get_text() for page in doc])
                    text_parts.append(f"--- Extracted PDF Content ---\n{extracted}\n--- End PDF Content ---")
            elif isinstance(part, str):
                text_parts.append(part)
            else:
                text_parts.append(str(part))

        full_text = "\n\n".join(text_parts)
        system_prompt = None
        configs_for_openai = dict(generation_configs) if generation_configs else {}
        if "system_instruction" in configs_for_openai:
            system_prompt = configs_for_openai.pop("system_instruction")
        if "temperature" in configs_for_openai:
            configs_for_openai.pop("temperature")

        return call_openai_models_with_text_prompt(
            prompt=full_text,
            model_name=model_name,
            generation_configs=configs_for_openai,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
            system_prompt=system_prompt,
        )
    else:
        return call_gemini_with_contents(
            contents=contents,
            model_name=model_name,
            generation_configs=generation_configs,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
        )
