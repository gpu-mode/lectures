# Supplementary Material for Lectures 2 & 3

## Lecture 2

- Recap Ch. 1-3 from the PMPP book
- Date: 2024-01-20, Speaker: [Andreas Koepf](https://twitter.com/neurosp1ke), Book: [Programming Massively Parallel Processors: A Hands-on Approach](https://a.co/d/2S2fVzt) (Amazon link)
- Slides: The powerpoint file [cuda_mode_lecture2.pptx](./cuda_mode_lecture2.pptx) can be found in the root directory of this repository. Alternatively [here](https://docs.google.com/presentation/d/1deqvEHdqEC4LHUpStO6z3TT77Dt84fNAvTIAxBJgDck/edit#slide=id.g2b1444253e5_1_75) as Google docs presentation.
- Examples: Please make sure PyTorch (2.1.2) and cuda-toolkit (nvcc compiler) are installed.
  - `vector_addition`: Classic CUDA C example, to compile use `make` in the `vector_addition` directory.
  - `rgb_to_grayscale`: Example uses PyTorch's `torch.utils.cpp_extension.load_inline` feature to compile a custom RGB to grayscale kernel and uses it to convert input image to grayscale and which is saved in as `output.png`. Run in the `rgb_to_grayscale` folder `python rgb_to_grayscale.py`.
  - `mean_filter`: This example also uses the PyTorch's `cpp_extension.load_inline` feature to compile a mean filter kernel. The kernel read pixel values in the surrounding (square area) of a pixel and computes the average value for each RGB channel individualy. The result is saved to `output.png`. Run in the `mean_filter` folder `python mean_filter.py`.

## Lecture 3

- Title: Getting Started With CUDA
- Date: 2024-01-27, Speaker: [Jeremy Howard](https://twitter.com/jeremyphoward)
- Notebook: See the `lecture3` folder, or run the [Colab version](https://colab.research.google.com/drive/180uk6frvMBeT4tywhhYXmz3PJaCIA_uk?usp=sharing)

