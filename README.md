# VideoGPT+ :movie_camera: :speech_balloon:

<p align="center">
  <img src="docs/images/videogpt_plus_face.jpeg" alt="videogpt_plus_face" width="200">
</p>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

### VideoGPT+: Integrating Image and Video Encoders for Enhanced Video Understanding

#### [Muhammad Maaz](https://www.muhammadmaaz.com) , [Hanoona Rasheed](https://www.hanoonarasheed.com/) , [Salman Khan](https://salman-h-khan.github.io/) and [Fahad Khan](https://sites.google.com/view/fahadkhans/home)

#### **Mohamed bin Zayed University of Artificial Intelligence**

---

[![paper](https://img.shields.io/badge/arXiv-Paper-blue.svg)](https://arxiv.org/abs/2406.09418)
[![video](https://img.shields.io/badge/Project-HuggingFace-F9D371)](https://huggingface.co/collections/MBZUAI/videogpt-665c8643221dda4987a67d8d)
[![Dataset](https://img.shields.io/badge/VCGBench-Diverse-green)](https://huggingface.co/datasets/MBZUAI/VCGBench-Diverse)
[![Demo](https://img.shields.io/badge/Annotation-Pipeline-red)](https://huggingface.co/datasets/MBZUAI/video_annotation_pipeline)

---
**Diverse Video-based Generative Performance Benchmarking (VCGBench-Diverse)**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videogpt-integrating-image-and-video-encoders/vcgbench-diverse-on-videoinstruct)](https://paperswithcode.com/sota/vcgbench-diverse-on-videoinstruct?p=videogpt-integrating-image-and-video-encoders)

**Video Question Answering on MVBench**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videogpt-integrating-image-and-video-encoders/video-question-answering-on-mvbench)](https://paperswithcode.com/sota/video-question-answering-on-mvbench?p=videogpt-integrating-image-and-video-encoders)


**Video-based Generative Performance Benchmarking**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videogpt-integrating-image-and-video-encoders/video-based-generative-performance)](https://paperswithcode.com/sota/video-based-generative-performance?p=videogpt-integrating-image-and-video-encoders)

---

## :loudspeaker: Latest Updates
- **Mar-28-25**: *Mobile-VideoGPT* is released. It achieves excellent results on multiple benchmarks with 2x higher throughput. Check it out [Mobile-VideoGPT](https://github.com/Amshaker/Mobile-VideoGPT) :fire::fire:

- **Jun-13-24**: VideoGPT+ paper, code, model, dataset and benchmark is released. :fire::fire:
---

## VideoGPT+ Overview :bulb:

VideoGPT+ integrates image and video encoders to leverage detailed spatial understanding and global temporal context, respectively. It processes videos in segments using adaptive pooling on features from both encoders, enhancing performance across various video benchmarks.

<p align="center">
  <img src="docs/images/block_diagram.png" alt="VideoGPT+ Architectural Overview">
</p>

---

## Contributions :trophy:

- **VideoGPT+ Model**: We present VideoGPT+, the first video-conversation model that benefits from a dual-encoding scheme based on both image and video features. These complimentary sets of features offer rich spatiotemporal details for improved video understanding.
- **VCG+ 112K Dataset**: Addressing the limitations of the existing VideoInstruct100K dataset, we develop VCG+ 112K with a novel semi-automatic annotation pipeline, offering dense video captions along with spatial understanding and reasoning-based QA pairs, further improving the model performance.
- **VCGBench-Diverse Benchmark**: Recognizing the lack of diverse benchmarks for video-conversation tasks, we propose VCGBench-Diverse, which provides 4,354 human annotated QA pairs across 18 video categories to extensively evaluate the performance of a video-conversation model.

<p align="center">
  <img src="docs/images/intro_radar_plot.png" alt="Contributions" width="650">
</p>

---

## Video Annotation Pipeline (VCG+ 112K) :open_file_folder:
Video-ChatGPT introduces the VideoInstruct100K dataset, which employs a semi-automatic annotation pipeline to generate 75K instruction-tuning QA pairs. To address the limitations of this annotation process, we present \ourdata~dataset developed through an improved annotation pipeline. Our approach improves the accuracy and quality of instruction tuning pairs by improving keyframe extraction, leveraging SoTA large multimodal models (LMMs) for detailed descriptions, and refining the instruction generation strategy.

<p align="center">
  <img src="docs/images/vcg120k_block_diagram.png" alt="Contributions">
</p>

---
## VCGBench-Diverse :mag:
Recognizing the limited diversity in existing video conversation benchmarks, we introduce VCGBench-Diverse to comprehensively evaluate the generalization ability of video LMMs. While VCG-Bench provides an extensive evaluation protocol, it is limited to videos from the ActivityNet200 dataset. Our benchmark comprises a total of 877 videos, 18 broad video categories and 4,354 QA pairs, ensuring a robust evaluation framework.

<p align="center">
  <img src="docs/images/vcgbench_block_diag.png" alt="Contributions">
</p>

---

## Installation :wrench:

We recommend setting up a conda environment for the project:
```shell
conda create --name=videogpt_plus python=3.11
conda activate videogpt_plus

git clone https://github.com/mbzuai-oryx/VideoGPT-plus
cd VideoGPT-plus

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.41.0

pip install -r requirements.txt

export PYTHONPATH="./:$PYTHONPATH"
```
Additionally, install [FlashAttention](https://github.com/HazyResearch/flash-attention) for training,
```shell
pip install ninja

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
python setup.py install
```
---

## Quantitative Evaluation 📊
We provide instructions to reproduce VideoGPT+ results on VCGBench, VCGBench-Diverse and MVBench. Please follow the instructions at [eval/README.md](eval/README.md).

### VCGBench Evaluation: Video-based Generative Performance Benchmarking :chart_with_upwards_trend:
<p align="center">
  <img src="docs/images/VCGBench_quantitative.png" alt="VCGBench_quantitative" width="1000">
</p>

---
### VCGBench-Diverse Evaluation :bar_chart:
<p align="center">
  <img src="docs/images/VCGDiverse_quantitative.png" alt="VCGDiverse_quantitative">
</p>

---
### Zero-Shot Question-Answer Evaluation :question:
<p align="center">
  <img src="docs/images/zero_shot_quantitative.png" alt="zero_shot_quantitative">
</p>

---

### MVBench Evaluation :movie_camera:
<p align="center">
  <img src="docs/images/MVBench_quantitative.png" alt="MVBench_quantitative">
</p>

---

## Training :train:
We provide scripts for pretraining and finetuning of VideoGPT+. Please follow the instructions at [scripts/README.md](scripts/README.md).

---

## Qualitative Analysis :mag:
A comprehensive evaluation of VideoGPT+ performance across multiple tasks and domains.
<p align="center">
  <img src="docs/images/demo_vcg+_main.png" alt="demo_vcg+_main" width="700">
</p>

---

<p align="center">
  <img src="docs/images/demo_vcg+_full_part1.jpg" alt="demo_vcg+_full_part1" width="700">
</p>


<p align="center">
  <img src="docs/images/demo_vcg+_full_part2.jpg" alt="demo_vcg+_full_part2" width="700">
</p>

---

## Acknowledgements :pray:

+ [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT): A pioneering attempt in Video-based conversation models.
+ [LLaVA](https://github.com/haotian-liu/LLaVA): Our code base is build upon LLaVA and Video-ChatGPT.
+ [Chat-UniVi](https://github.com/PKU-YuanGroup/Chat-UniVi): A recent work in image and video-based conversation models. We borrowed some implementation details from their public codebase.

## Citations 📜:

If you're using VideoGPT+ in your research or applications, please cite using this BibTeX:
```bibtex
@article{Maaz2024VideoGPT+,
    title={VideoGPT+: Integrating Image and Video Encoders for Enhanced Video Understanding},
    author={Maaz, Muhammad and Rasheed, Hanoona and Khan, Salman and Khan, Fahad Shahbaz},
    journal={arxiv},
    year={2024},
    url={https://arxiv.org/abs/2406.09418}
}

@inproceedings{Maaz2023VideoChatGPT,
    title={Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models},
    author={Maaz, Muhammad and Rasheed, Hanoona and Khan, Salman and Khan, Fahad Shahbaz},
    booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)},
    year={2024}
}
```

## License :scroll:
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.


Looking forward to your feedback, contributions, and stars! :star2:
Please raise any issues or questions [here](https://github.com/mbzuai-oryx/VideoGPT-plus/issues). 


---
[<img src="docs/images/IVAL_logo.png" width="200" height="100">](https://www.ival-mbzuai.com)
[<img src="docs/images/Oryx_logo.png" width="100" height="100">](https://github.com/mbzuai-oryx)
[<img src="docs/images/MBZUAI_logo.png" width="360" height="85">](https://mbzuai.ac.ae)
