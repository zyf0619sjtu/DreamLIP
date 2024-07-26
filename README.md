# DreamLIP: Language-Image Pre-training with Long Captions


> **DreamLIP: Language-Image Pre-training with Long Captions** <br>
<a href="https://zkcys001.github.io/">Kecheng Zheng</a>,</span>
<a href="https://github.com/zyf0619sjtu">Yifei Zhang</a>,</span>
<a href="https://github.com/wuw2019">Wei Wu</a>,</span>
<a href="https://github.com/LuFan31">Fan Lu</a>,</span>
<a href="https://scholar.google.com/citations?user=dNhzCu4AAAAJ&hl=zh-CN">Shuailei Ma</a>,</span>
<a href="http://home.ustc.edu.cn/~jinxustc/">Xin Jin</a>,</span>
<a href="http://www.cad.zju.edu.cn/home/chenwei/">Wei Chen</a>,</span>
<a href="https://shenyujun.github.io/">Yujun Shen</a> <br>
[Project Page](https://zyf0619sjtu.github.io/dream-lip/) | [Paper](https://arxiv.org/pdf/2403.17007.pdf) | [Data](https://drive.google.com/file/d/19jCNWvy7kA70u-ufQtEJvbKVMG2b8MnP/view?usp=drive_link)


## 📰 News

- [2024/07/26] Long captions (LLAVA1.5, InstructBLIP and shareGPT4V) of CC3M and CC12M are released~
- [2024/07/16] Upload the pretrained weight of VIT-B/16 pretrained in CC3M, CC12M, YFCC15M, and merged-30M (long captions of ShareGPT4V)!
- [2024/07/08] DreamLIP is accepted by ECCV 2024!

## 💡 Highlights
- 🔥 Exploring how language-image pre-training could benefit from long captions.
- 🔥 Strong improvement on semantic segmentation, image-text retrieval, semantic segmentation, and image understanding in MLLM.

<img src="figures/radar.jpg" style="vertical-align: -10px; display: block; margin-left: auto; margin-right: auto;" height="400px" width="440px">

- 🔥 DreamLIP trained with 30M image-text pairs achieves on par or even better performance than CLIP trained with 400M pairs.
![timeline.jpg](figures/moti.png)

## 🎨 In-Progress

- [ ] Release long captions of YFCC15M.
- [ ] Release training code

## 🏝️ Overview of supported long captions:

<details open>
<summary><b>Long Captions of Supported Datasets (5)</b></summary>

> - [x] [![](https://img.shields.io/badge/CC3M-red?style=for-the-badge)](https://ai.google.com/research/ConceptualCaptions/)
> - [x] [![](https://img.shields.io/badge/CC12M-d0e9ff?style=for-the-badge)](https://github.com/google-research-datasets/conceptual-12m)
> - [x] [![](https://img.shields.io/badge/YFCC15M-yellowgreen?style=for-the-badge)](https://github.com/Sense-GVT/DeCLIP/blob/main/docs/dataset_prepare.md)
> - [ ] [![](https://img.shields.io/badge/Laion-grey?style=for-the-badge)](https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/)
> - [ ] [![](https://img.shields.io/badge/Coyo-854?style=for-the-badge)](https://github.com/kakaobrain/coyo-dataset)
</details>
<details open>
<summary><b>Long Captions of MLLMs (3)</b></summary>

> - [x] ![](https://img.shields.io/badge/InstructBLIP-blue?style=for-the-badge) 
> - [x] ![](https://img.shields.io/badge/LLAVA1.5-green?style=for-the-badge) 
> - [x] ![](https://img.shields.io/badge/SHAREGPT4V-orange?style=for-the-badge) 

</details>

[//]: # (## Acknowledgement)


#### Generated Long Captions

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Raw/Long/Short Caption</th>
<th valign="center">InstructBLIP + LLAVA1.5 + ShareGPT4V</th>

<!-- TABLE BODY -->
<tr>
<td align="center">CC3M</td>
<td align="center"><a href="https://drive.google.com/file/d/1RPcFS8jrVolA9RzHXD581E8BxR7jYDap/view?usp=sharing">Link</a></td>
</tr>
<tr>
<td align="center">CC12M</td>
<td align="center"><a href="https://drive.google.com/file/d/12iUhceznPNWd-l_bGSF5rSnzdruP4Jtr/view?usp=sharing">Link</a></td>
</tr>
<tr>
<td align="center">YFCC15M</td>
<td align="center">TODO</td>
</tr>
</tbody></table>

## Pretrained checkpoints
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Model</th>
<th valign="center">ShareGPT4V</th>
<th valign="center">InstructBLIP + LLAVA1.5 + ShareGPT4V</th>

<!-- TABLE BODY -->
<tr>
<td align="center">CC3M</td>
<td align="center">ViT-B/16</td>
<td align="center"><a href="https://drive.google.com/file/d/1f8JdXmdGRQtCzXpEGTpE_T7bWXLMnaMj/view?usp=sharing">Link</a></td>
<td align="center">TODO</td>
</tr>
<tr>
<td align="center">CC12M</td>
<td align="center">ViT-B/16</td>
<td align="center"><a href="https://drive.google.com/file/d/12qSRzW8q2Jg2L4y05s-AMXyCvPS7O6BK/view?usp=sharing">Link</a></td>
<td align="center">TODO</td>
</tr>
<tr>
<td align="center">YFCC15M</td>
<td align="center">ViT-B/16</td>
<td align="center"><a href="https://drive.google.com/file/d/1CG1-XRsnff7b26WYdygNOWnhAqI5y_a7/view?usp=sharing">Link</a></td>
<td align="center">TODO</td>
</tr>
<tr>
<td align="center">CC30M</td>
<td align="center">ViT-B/16</td>
<td align="center"><a href="https://drive.google.com/file/d/1pPVVOt_YALq_YX7x2kNEfDWSdHQ5wqew/view?usp=sharing">Link</a></td>
<td align="center">TODO</td>
</tr>
</tbody></table>

## 📣 Instructions
Environment installation
```
pip install -r requirments.txt
```

Evaluate zero shot classification
```
bash eval_zs.sh
```

[//]: # (You can download checkpoints pre-trained )



## 📖 Citation

```bibtex
@inproceedings{DreamLIP,
  title={DreamLIP: Language-Image Pre-training with Long Captions},
  author={Zheng, Kecheng and Zhang, Yifei and Wu, Wei and Lu, Fan and Ma, Shuailei and Jin, Xin and Chen, Wei and Shen, Yujun},
  booktitle={ECCV},
  year={2024}
}
```

### Acknowledgements
This project is based on [open_clip](https://github.com/mlfoundations/open_clip/tree/main), and thanks for the nice work! 
We also thank [InstructBLIP](https://github.com/salesforce/LAVIS), [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer) and [LLAVA](https://github.com/haotian-liu/LLaVA) for the pretrained models and codes.
