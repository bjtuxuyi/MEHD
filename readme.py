"""
This is the pytorch implement of paper "Multivariate event hypergraph diffusion model for train delay prediction"
We refer to the code implementation of DSTPP[https://doi.org/10.1145/3580305.3599511] in our work, and we hereby express our sincere gratitude.

Abstract:
Train delay prediction is a key technology for train scheduling and timetable optimization, and constitutes a critical
component of intelligent transportation systems. We present the first study on regional-level multi-train delay prediction
problem, and focus on modeling the regional-level delay propagation and evolution process, and capturing coordinated
operation status among multiple train clusters in the complex operation network. First, we propose a brand-new Multivariate
 Event Hypergraph Diffusion (MEHD) model, and introduce a novel data structure, the mixed hypergraph, which accurately
 models the spatio-temporal high-order correlations between the regional-level multi-train arrival events. Then, we propose
 a mixed hypergraph convolution method to characterize complex train operation network, which improves the ability to capture
  the spatio-temporal high-order correlations and non-Euclidean characteristics between events. Finally, we propose an event
   hypergraph diffusion process, and design a prior operational schedule-conditioned attention denoising module to enhance
   the ability to learn all train arrival event generation mechanisms. Extensive experiments demonstrate that our MEHD
   achieves superior performance compared to current state-of-the-art models on actual high-speed rail performance datasets,
   with an average improvement of 20%-30% on multiple metrics, and performs good robustness and efficiency. Subsequent
   experiments and analyses demonstrate the unique advantages of MEHD over single-train prediction methods. To the best
   of our knowledge, this is the first end-to-end model for regional-level multi-train delay prediction.

paper link: https://doi.org/10.1016/j.trc.2025.105390

program entry:python MySPTT/HY/main_HGConv.py --Optional parameters are available in MySPTT/utils/setup_utils.py

If you find our work helpful, please cite our paper.
@article{xu2026multivariate,
  title={Multivariate event hypergraph diffusion model for train delay prediction},
  author={Xu, Yi and Li, Honghui and Wu, Chang and Peng, Yunjuan and Du, Xilu and Wang, Hongwei and Mohammed, Sabah and Calvi, Alessandro and Zhang, Dalin},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={182},
  pages={105390},
  year={2026},
  publisher={Elsevier}
}

"""


















