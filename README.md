# (D)RAM (Deep Recurrent Attention Model)

Implementation of RAM (Mnih et al. 2014) and different versions of DRAM (Ba et al. 2015). The models are tested on original and modified versions of MNIST and SVHN datasets.

3 versions of DRAM are implemented. 
1st version has a simple context and glimpse networks optimized for MNIST addition task
2nd version has more complicated context and glimpse networks that are optimized for MNIST addition task
3rd version same as the 2nd version except optimized for multidigit classification of SVHN.

References

Williams, R. J. (1992). Simple statistical gradient-following algorithms for con-nectionist reinforcement learning.Machine learning,8(3-4), 229–256.

Goodfellow, I. J., Bulatov, Y., Ibarz, J., Arnoud, S., & Shet, V. (2013). Multi-digit number recognition from street view imagery using deep convolu-tional neural networks.arXiv preprint arXiv:1312.6082.

Ba, J., Mnih, V., & Kavukcuoglu, K. (2014). Multiple object recognition withvisual attention.arXiv preprint arXiv:1412.7755.

Mnih, V., Heess, N., Graves, A., & Kavukcuoglu, K. (2014). Recurrent modelsof visual attention.arXiv preprint arXiv:1406.6247.

Ablavatski, A., Lu, S., & Cai, J. (2017). Enriched deep recurrent visual attentionmodel for multiple object recognition.2017 IEEE Winter Conferenceon Applications of Computer Vision (WACV), 971–978

Kesen I, RAM.jl (2019). https://github.com/ilkerkesen/RAM
