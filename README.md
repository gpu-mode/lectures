# Supplementary Material for Lectures

The PMPP Book: [Programming Massively Parallel Processors: A Hands-on Approach](https://a.co/d/2S2fVzt) (Amazon link)


## Lecture 1

- Title: Profiling and Integrating CUDA kernels in PyTorch
- [Video](https://youtu.be/LuhJEEJQgUM)
- Date: 2024-01-13, Speaker: [Mark Saroufim](https://twitter.com/marksaroufim)
- Notebook and slides in `lecture1` folder


## Lecture 2

- Title: Recap Ch. 1-3 from the PMPP book
- [Video](https://youtu.be/NQ-0D5Ti2dc)
- Date: 2024-01-20, Speaker: [Andreas Koepf](https://twitter.com/neurosp1ke)
- Slides: The powerpoint file [lecture2/cuda_mode_lecture2.pptx](./lecture2/cuda_mode_lecture2.pptx) can be found in the root directory of this repository. Alternatively [here](https://docs.google.com/presentation/d/1deqvEHdqEC4LHUpStO6z3TT77Dt84fNAvTIAxBJgDck/edit#slide=id.g2b1444253e5_1_75) as Google docs presentation.
- Examples: Please make sure PyTorch (2.1.2) and cuda-toolkit (nvcc compiler) are installed.
  - `lecture2/vector_addition`: Classic CUDA C example, to compile use `make` in the `vector_addition` directory.
  - `lecture2/rgb_to_grayscale`: Example uses PyTorch's `torch.utils.cpp_extension.load_inline` feature to compile a custom RGB to grayscale kernel and uses it to convert input image to grayscale and which is saved in as `output.png`. Run in the `lecture2/rgb_to_grayscale` folder `python rgb_to_grayscale.py`.
  - `lecture2/mean_filter`: This example also uses the PyTorch's `cpp_extension.load_inline` feature to compile a mean filter kernel. The kernel read pixel values in the surrounding (square area) of a pixel and computes the average value for each RGB channel individualy. The result is saved to `output.png`. Run in the `lecture2/mean_filter` folder `python mean_filter.py`.


## Lecture 3

- Title: Getting Started With CUDA
- [Video](https://youtu.be/4sgKnKbR-WE)
- Date: 2024-01-27, Speaker: [Jeremy Howard](https://twitter.com/jeremyphoward)
- Notebook: See the `lecture3` folder, or run the [Colab version](https://colab.research.google.com/drive/180uk6frvMBeT4tywhhYXmz3PJaCIA_uk?usp=sharing)


## Lecture 4

- Title: Intro to Compute and Memory Architecture
- [Video](https://youtu.be/lTmYrKwjSOU)
- Date: 2024-02-03, Speaker: [Thomas Viehmann](https://lernapparat.de/)
- Notebook and slides in the `lecture4` folder.


## Lecture 5

- Title: Going Further with CUDA for Python Programmers
- [Video](https://youtu.be/wVsR-YhaHlM)
- Date: 2024-02-10, Speaker: [Jeremy Howard](https://twitter.com/jeremyphoward)
- Notebook in the `lecture5` folder.
