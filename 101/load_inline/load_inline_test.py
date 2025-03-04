import torch 
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world() {
return {"hello world"};
}
"""

new_module = load_inline(
    name="new_module",
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True,
    build_directory="./build"
)

print(new_module.hello_world())