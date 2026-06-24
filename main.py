import sys
import time

# Conceptual GPU HBM Capacity (e.g., 16 GB for a mid-range AI GPU)
HBM_CAPACITY_GB = 16
HBM_CAPACITY_BYTES = HBM_CAPACITY_GB * (1024**3)

print(f"Simulating GPU HBM Capacity: {HBM_CAPACITY_GB} GB\n")

current_hbm_usage_bytes = 0
loaded_data_objects = [] # Keep references to prevent garbage collection

def allocate_and_report_memory(name, size_gb):
    """
    Simulates allocating memory for model weights or activations.
    Returns a bytearray representing the allocated data.
    """
    global current_hbm_usage_bytes
    size_bytes = int(size_gb * (1024**3))
    
    # Check if this allocation would exceed our conceptual HBM limit
    if current_hbm_usage_bytes + size_bytes > HBM_CAPACITY_BYTES:
        print(f"--- WARNING: '{name}' allocation of {size_gb:.2f} GB would exceed HBM capacity! ---")
        print(f"    Current usage: {current_hbm_usage_bytes / (1024**3):.2f} GB, Remaining: {(HBM_CAPACITY_BYTES - current_hbm_usage_bytes) / (1024**3):.2f} GB")
        return None

    print(f"Allocating '{name}' data: {size_gb:.2f} GB...")
    try:
        # Using bytearray to represent raw memory allocation, mimicking large tensors
        data = bytearray(size_bytes)
        # sys.getsizeof includes object overhead, giving a more realistic footprint
        current_hbm_usage_bytes += sys.getsizeof(data)
        loaded_data_objects.append(data) # Store reference
        print(f"    Allocated {sys.getsizeof(data) / (1024**3):.2f} GB. Total HBM usage: {current_hbm_usage_bytes / (1024**3):.2f} GB / {HBM_CAPACITY_GB:.2f} GB")
        return data
    except MemoryError:
        print(f"    Failed to allocate {size_gb:.2f} GB for '{name}'. System MemoryError.")
        return None

def simulate_data_processing(data_array, description, iterations=1000000):
    """
    Simulates processing data in memory to hint at bandwidth usage.
    (This is a CPU-bound simulation, not actual HBM bandwidth, but illustrates
    that more data generally means more work and time).
    """
    if data_array is None:
        print(f"Cannot process '{description}': Data not allocated.")
        return

    print(f"\nSimulating processing for '{description}' ({sys.getsizeof(data_array) / (1024**3):.2f} GB)...")
    start_time = time.perf_counter()
    
    # Simulate reading/writing data. Accessing elements in a loop.
    # This loop is simple but demonstrates that operations on larger data
    # will inherently take more time, analogous to HBM bandwidth pressure.
    data_len = len(data_array)
    if data_len > 0:
        for i in range(iterations):
            # Access a few elements to simulate data operations (e.g., matrix multiplication, attention)
            idx1 = (i * 17) % data_len
            idx2 = (i * 31) % data_len
            _ = data_array[idx1] + data_array[idx2] # Simulate reading
            data_array[idx1] = (data_array[idx1] + 1) % 256 # Simulate writing (mod 256 for byte)
    
    end_time = time.perf_counter()
    print(f"    Processing finished in {end_time - start_time:.4f} seconds. (Simulated operations: {iterations})")

# --- Scenario 1: Loading a large Vision Encoder Model --- 
# Vision models often require large intermediate activation tensors for high-resolution inputs.
print("\n--- Scenario 1: Loading a Large Vision Encoder Model ---")
vision_model_weights = allocate_and_report_memory("Vision Encoder Weights", 8.0) # e.g., 8 GB for a large CNN/Transformer
vision_activations_batch1 = allocate_and_report_memory("Vision Activations (Batch 1)", 2.0) # e.g., 2 GB for a batch of high-res images

# --- Scenario 2: Loading a large Language Decoder Model --- 
# LLMs are known for massive weight sizes and large key/value caches for long contexts.
print("\n--- Scenario 2: Loading a Large Language Decoder Model ---")
llm_model_weights = allocate_and_report_memory("Language Decoder Weights", 10.0) # e.g., 10 GB for a 7B parameter LLM
llm_activations_batch1 = allocate_and_report_memory("Language Decoder Activations (Batch 1)", 3.0) # e.g., 3 GB for a long text sequence batch

# --- Scenario 3: Competition for HBM (trying to run both simultaneously) ---
# This demonstrates the 'HBM tax' where total memory demand exceeds capacity.
print("\n--- Scenario 3: Competition for HBM (Attempting to run both concurrently) ---")
# Reset usage for a fresh scenario to clearly demonstrate competition
current_hbm_usage_bytes = 0
loaded_data_objects = []

print("Attempting to load Vision Encoder and Language Decoder models concurrently...")
vision_model_weights_comp = allocate_and_report_memory("Vision Encoder Weights (Concurrent)", 8.0)
# This allocation will likely fail or warn about exceeding capacity if the vision model was loaded first
llm_model_weights_comp = allocate_and_report_memory("Language Decoder Weights (Concurrent)", 10.0)

# --- Scenario 4: Illustrating "HBM Tax" through Processing Time --- 
# Even if memory fits, processing larger data implies higher bandwidth demands, leading to longer times.
print("\n--- Scenario 4: Illustrating 'HBM Tax' through Processing Time ---")
# Reset for this scenario
current_hbm_usage_bytes = 0
loaded_data_objects = []

small_data = allocate_and_report_memory("Small Data (1 GB)", 1.0)
large_data = allocate_and_report_memory("Large Data (8 GB)", 8.0)

# Simulate processing for both, with the same number of conceptual operations
simulate_data_processing(small_data, "Small Data (1 GB)", iterations=500000)
simulate_data_processing(large_data, "Large Data (8 GB)", iterations=500000) # Larger data takes longer for same operations

print("\n--- Simulation Complete ---")
