"""OpenAI embedding generation with batching and retry logic."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

import numpy as np
from openai import OpenAI
from tqdm import tqdm

import config


def get_openai_client() -> OpenAI:
    """Initialize and return the OpenAI-compatible client."""
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_EMBEDDING_BASE_URL,
    )


def embed_texts(
    texts: Sequence[str],
    client: OpenAI | None = None,
    model: str = config.EMBEDDING_MODEL,
    dimensions: int = config.EMBEDDING_DIMENSION,
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
    num_workers: int = config.EMBEDDING_NUM_WORKERS,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate embeddings for a list of texts using OpenAI API with multithreading.
    
    Args:
        texts: List of texts to embed.
        client: OpenAI client. If None, creates a new one.
        model: Embedding model to use.
        dimensions: Output embedding dimension (for models that support it).
        batch_size: Number of texts to embed per API call.
        num_workers: Number of concurrent threads for parallel processing.
        show_progress: Whether to show a progress bar.
    
    Returns:
        Tuple of:
        - numpy array of shape (len(texts), embedding_dimension)
        - list of dicts containing info about skipped texts (index, text, reason)
    """
    if client is None:
        client = get_openai_client()
    
    texts = list(texts)  # Ensure it's a list
    
    # Prepare batches with their indices
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batches.append((i, batch))
    
    num_batches = len(batches)
    results = [None] * num_batches  # Pre-allocate to maintain order
    skipped_texts = []  # Track texts that were skipped
    
    def process_batch(batch_info: tuple[int, int, list[str]]) -> tuple[int, list[list[float]], list[dict]]:
        """Process a single batch and return (batch_index, embeddings, skipped_info)."""
        batch_idx, start_idx, batch = batch_info
        embeddings, batch_skipped = _embed_batch_with_retry(batch, client, model, dimensions, start_idx)
        return batch_idx, embeddings, batch_skipped
    
    # Submit all tasks to thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create tasks with batch index for ordering
        tasks = [
            (batch_idx, start_idx, batch) 
            for batch_idx, (start_idx, batch) in enumerate(batches)
        ]
        futures = {executor.submit(process_batch, task): task[0] for task in tasks}
        
        # Use tqdm for progress tracking
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=num_batches,
                desc="Generating embeddings",
            )
        
        for future in iterator:
            batch_idx = futures[future]
            try:
                _, embeddings, batch_skipped = future.result()
                results[batch_idx] = embeddings
                skipped_texts.extend(batch_skipped)
            except Exception as e:
                raise RuntimeError(f"Failed to process batch {batch_idx}: {e}")
    
    # Flatten results while maintaining order
    all_embeddings = []
    for batch_embeddings in results:
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings, dtype=np.float32), skipped_texts


def _embed_batch_with_retry(
    texts: list[str],
    client: OpenAI,
    model: str,
    dimensions: int,
    start_idx: int = 0,
) -> tuple[list[list[float]], list[dict]]:
    """
    Embed a batch of texts with retry logic for rate limits and errors.
    
    Returns:
        Tuple of (embeddings, skipped_texts_info)
    """
    skipped_texts = []
    
    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model,
                dimensions=dimensions,
            )
            return [item.embedding for item in response.data], skipped_texts
        
        except Exception as e:
            error_str = str(e)
            
            # Check if this is a token length error (400 error with "maximum context length")
            if "400" in error_str and "maximum context length" in error_str:
                print(f"\n⚠ Token length error detected. Processing texts individually...")
                
                # Process each text individually
                embeddings = []
                for idx, text in enumerate(texts):
                    global_idx = start_idx + idx
                    try:
                        response = client.embeddings.create(
                            input=[text],
                            model=model,
                            dimensions=dimensions,
                        )
                        embeddings.append(response.data[0].embedding)
                    except Exception as individual_e:
                        individual_error_str = str(individual_e)
                        if "400" in individual_error_str and "maximum context length" in individual_error_str:
                            # This individual text is too long, use random embedding
                            print(f"  ⚠ Skipping text {global_idx} (too long)")
                            random_embedding = np.random.randn(dimensions).tolist()
                            embeddings.append(random_embedding)
                            
                            # Track the skipped text
                            skipped_texts.append({
                                "index": global_idx,
                                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                                "text_length": len(text),
                                "reason": f"Token length exceeded: {individual_error_str}",
                            })
                        else:
                            # Some other error, re-raise
                            raise individual_e
                
                return embeddings, skipped_texts
            
            # Not a token length error, use normal retry logic
            if attempt < config.MAX_RETRIES - 1:
                wait_time = config.RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to embed batch after {config.MAX_RETRIES} attempts: {e}")


def embed_single(
    text: str,
    client: OpenAI | None = None,
    model: str = config.EMBEDDING_MODEL,
    dimensions: int = config.EMBEDDING_DIMENSION,
) -> np.ndarray:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text to embed.
        client: OpenAI client. If None, creates a new one.
        model: Embedding model to use.
        dimensions: Output embedding dimension (for models that support it).
    
    Returns:
        numpy array of shape (embedding_dimension,)
    """
    if client is None:
        client = get_openai_client()
    
    response = client.embeddings.create(
        input=[text],
        model=model,
        dimensions=dimensions,
    )
    
    return np.array(response.data[0].embedding, dtype=np.float32)
