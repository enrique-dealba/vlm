"""FastAPI server for handling Large Language Model (LLM) requests."""

import logging
import os

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain.llms import VLLM
from pydantic import BaseModel, ValidationError

from config import Settings

settings = Settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUNotAvailableError(Exception):
    """Custom exception for when GPU is not available."""

    pass


class GenerateRequest(BaseModel):
    """Schema for LLM text generation request."""

    text: str


def get_quantization():
    """Returns quantization method based on config settings."""
    quantization = os.environ.get("QUANTIZATION", "None")
    if "GPTQ" in settings.DEFAULT_MODEL:
        quantization = "gptq"
    elif "AWQ" in settings.DEFAULT_MODEL:
        quantization = "awq"
    else:
        quantization = None
    return quantization


def create_llm() -> VLLM:
    """Creates and returns VLLM instance based on current configuration."""
    if not torch.cuda.is_available():
        raise GPUNotAvailableError("No GPU available. VLLM requires GPU acceleration.")

    quantization = get_quantization()
    if quantization is None:
        gpu_utilization = settings.DEFAULT_GPU_UTIL
        dtype_value = "bfloat16"
    else:
        gpu_utilization = getattr(
            settings, f"{quantization.upper()}_GPU_UTIL", settings.DEFAULT_GPU_UTIL
        )
        dtype_value = "half" if quantization in ["awq", "gptq"] else "bfloat16"

    try:
        llm = VLLM(
            model=settings.DEFAULT_MODEL,
            temperature=settings.TEMPERATURE,
            use_beam_search=False,
            max_new_tokens=settings.MAX_TOKENS,
            tensor_parallel_size=settings.NUM_GPUS,
            trust_remote_code=False,
            dtype=dtype_value,
            vllm_kwargs={
                "quantization": quantization,
                "gpu_memory_utilization": gpu_utilization,
                # "max_model_len": settings.MAX_SEQ_LEN,
            },
        )

        return llm
    except (ValidationError, RuntimeError) as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise GPUNotAvailableError(f"Failed to initialize LLM: {e}")


try:
    llm = create_llm()
except GPUNotAvailableError as e:
    logger.critical(f"Critical error: {e}")
    llm = None

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """Verify LLM initialization on server startup."""
    if llm is None:
        logger.critical("LLM failed to initialize. Server cannot start.")
        raise RuntimeError("LLM failed to initialize. Server cannot start.")


def get_llm():
    """Retrieves initialized LLM or raises exception if unavailable."""
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM service is unavailable due to GPU initialization failure.",
        )
    return llm


@app.post("/generate")
async def generate(request: Request, llm: VLLM = Depends(get_llm)):
    """Endpoint to generate text using LLM."""
    try:
        request_data = await request.json()
        query = request_data.get("text")
        if not query:
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        response = llm(query)
        return JSONResponse({"text": response})
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during text generation. Please try again later.",
        )


@app.get("/health")
async def health_check():
    """Check health status of the LLM service."""
    if llm is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "LLM is not initialized. GPU may not be available.",
            },
        )
    return JSONResponse(
        {"status": "healthy", "message": "LLM is initialized and ready."}
    )
