import torch
large_value = torch.tensor([1000.0], dtype=torch.float32)  # Using float32 for initial value

# Define a smaller value that is significant for float32 but not for float16
small_value = torch.tensor([1e-3], dtype=torch.float32)  # Small value in float32

# Add small value to large value in float32
result_float32 = large_value + small_value

# Convert large value to float16 and add the small value (also converted to float16)
result_float16 = large_value.to(torch.float16) + small_value.to(torch.float16)

# Convert results back to float32 for accurate comparison
result_float32 = result_float32.item()
result_float16_converted = result_float16.to(torch.float32).item()

# Print results
# 1000.0009765625 1000.0
print(result_float32, result_float16_converted)
