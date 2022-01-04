# Music-to-Dance Generation with Optimal Transport

Sample Results for MDOT-Net

https://user-images.githubusercontent.com/97083760/148023174-4bfc8fbe-c8a4-4553-93b0-8a42ae89083a.mp4

Complete code and data will be released upon acceptance.



## Overview of our Approach

We employ optimal transport theory for our generative framework. It constitutes a general cross domain sequence to sequence learning setting.



## Motivations

Our rationale for adopting optimal transport theory are the following:

- In traditional GANs, the training process is often precarious and unstable (the objectives might become divergent during training), requiring serious efforts to tune for the right hyperparameters.
- The WGAN is subject to the [curse of dimensionality](https://arxiv.org/pdf/1707.00087.pdf). That is to say, the amount of data samples required for convergence grows exponentially with respect to the (representation) dimensions of the target domain.
- The entropy regularized optimal transport distance is not subject to these issues. The objectives are provably well-defined and non-divergent throughout the generator training. This mitigates the generator divergence and instability issues.
- Furthermore, we can enhance the source to target domain matching via the Gromov-Wasserstein distance.

## Related Works

### Optimal Transport for Generative modeling

Optimal transport defines a metric distance for probability distributions over arbitrary spaces. 

- The generative modeling problem is reframed as finding an optimized transport for aligning the model distribution and data distribution. 
- However, solving the optimal transport problem is expensive, and this computational burden presented major hurdles for employing optimal transport for generative modeling.
- The Wasserstein GAN turned to the dual optimal transport problem and proposed a viable means of approximating 1-Lipschitz functions for GAN training. 
- An alternative line of work was pursued in [Sinkhorn GAN](http://proceedings.mlr.press/v84/genevay18a/genevay18a.pdf), [OT-GAN](https://arxiv.org/pdf/1803.05573), in which the introduction of an entropic regularization term reduced the computational cost. The regularized primal optimal transport problem is amenable to backpropagation training.

### Optimal Transport for Cross Domain Matching

The Gromov-Wasserstein distance generalizes the optimal transport for comparing distributions supported on different domains. and it is defined as a distance between intra-domain distances. 

- The Gromov-Wasserstein distance is promising for [learning cross-domain correspondences](http://proceedings.mlr.press/v97/bunne19a/bunne19a.pdf) such as for [unsupervised language translation](https://www.aclweb.org/anthology/D18-1214.pdf) or [graph matching](https://arxiv.org/pdf/1901.06003).



## Our Proposed Framework

Formally, let us consider a general task where we are interested in learning a sequence to sequence model across two domains. For example, if we are interested in a music to dance generation task, the source domain is the space of music, whereas the target domain would be the human pose manifold.

- Denote the **real** distribution over the *target domain* $\mathcal{T}$ as $\nu$.

- Denote the distribution over the *source domain* $\mathcal{S}$ as $\xi$.

- A generator network, with parameters $\theta$, learns a parametric mapping $$g_\theta$$ that maps an input sequence $y$ (sampled from $\xi$) to a generated sequence $\tilde{x}$. This gives the model distribution $\mu_{\theta}$ over the *target domain*.

- $\mu_{\theta}$ is formally defined as the push forward probability measure of $\xi$ under the generator mapping $g_{\theta}$, i.e. $\mu_\theta={g_{\theta_{\#}}(\xi)}$.

- Our idea consists of using an optimal transport distance $OT(\mu_{\theta},\nu)$ and a Gromov-Wasserstein distance $GW(\mu_{\theta},\xi)$ as objective functions to facilitate learning of the generator parameters $\theta$
  $$
  \arg\min_{\theta} \;OT (\mu_{\theta},\nu) + GW (\mu_{\theta},\xi).
  $$
  In what follows, we present the definitions for $OT (\mu_{\theta},\nu)$ and $GW (\mu_{\theta},\xi)$.

### Optimal Transport Distance $OT(\mu_\theta,\nu)$

Optimal transport defines a distance between the generated target distribution $\mu_{\theta}$ and the real target distribution $\nu$. It is defined as the solution of the following Kantorovich problem:
$$
OT_c (\mu_{\theta},\nu) =\inf_{\gamma \in \Gamma(\mu_{\theta},\nu)} \mathop{\mathbb{E}}_{(\tilde{x}, x) \sim \gamma}[c(\tilde{x},x)]
$$
where $\Gamma(\mu_{\theta},\nu)$ denotes the set of all joint distributions $\gamma$ whose marginals are $\mu_{\theta}$ and $\nu$. Intuitively, the above equation optimizes over all transport plans transforming $\mu_{\theta}$ into $\nu$ whereby the cost of moving a model sequence $\tilde{x}$ to data sequence $x$ is given by  $c(\tilde{x},x)$.

The cost function $c(\tilde{x},x)$ can be learnt by a network (this would correspond to the critic network in WGAN) or it can also be pre-defined according to the task. Personally, I would think that it is better to learn a cost function for image-based tasks whereas for pose/motion task, we can define the cost function as the geodesic distance over the human pose manifold.

- We can simplify the Kantorovich problem through the following. We consider that $\mu_{\theta}$ comprises $m$ generated sequences $\{\tilde{x}_i\}_{i=1}^m$ and $\nu$ comprises $m$ data sequences $\{x_j\}_{j=1}^m$. $\mu_{\theta}$ and $\nu$ may thus be viewed as discrete distributions $\mu_{\theta}=\frac{1}{m}\sum_{i=1}^m\delta_{\tilde{x}_i}$, $\nu=\frac{1}{m}\sum_{j=1}^m\delta_{x_j}$ where $\delta$ denotes the Dirac delta distribution. This discretization reduces the transport plans $\Gamma(\mu_{\theta},\nu)$ to
  $$
  \Gamma=\left\{\gamma\in\mathbb{R}^{m\times m}_+\mid\forall{i}\sum_{j}\gamma_{ij}=1,\forall{j}\sum_{i}\gamma_{ij}=1 \right\},
  $$
  i.e., a $m\times m$ matrix with positive entries such that every row or column sums to 1.

- The OT distance is then defined via the following optimization problem
  $$
  OT_c (\mu_{\theta},\nu)=\min_{\gamma \in \Gamma} \sum_{i,j} \gamma_{ij}c(\tilde{x}_i,x_j).
  $$
  This means we optimize over all possible transport plan $\gamma$ matrices with respect to the cost of moving each model sequence $\tilde{x}_i$ to data sequence $x_j$.

- We also need to introduce an entropy regularization term.
  $$
  OT_{c,\epsilon}(\mu_{\theta},\nu) =\min_{\gamma \in \Gamma}\sum_{i=1}^n\sum_{j=1}^n \gamma_{ij}c(\tilde{x}_i,x_j)+\epsilon I(\gamma)
  $$
  where $I(\gamma) = \sum_{i,j}\gamma_{ij}\log_2\gamma_{ij}$ is the mutual information of $\mu_{\theta},\nu$.

- This regularization transforms the Kantorovich problem into a convex optimization problem which can be efficiently solved via the following algorithm.

#### Computing the OT Distance for batch of $m$ samples with Sinkhorn-Knopp algorithm

**Input**: generated sequences $\widetilde{\mathbf{X}}=\{\tilde{x}_i\}_{i=1}^m$

**Input**: data sequences $\mathbf{X}=\{x_j\}_{j=1}^m$

**Hyperparameters**: regularization $\epsilon$, Sinkhorn iterations $L$

1. Compute Cost Matrix $C_{ij}=c(\tilde{x}_i,x_j)$.

2. $K = \exp(-C/\epsilon)$

3. Initialize $\mathbf{b}^{(0)}=\mathbb{1}_m$ where $\mathbb{1}_m=(1,\cdots,1)^T\in\mathbb{R}^m$ 

4. for $\ell = 1:L$

   ​	$\mathbf{a}^{(\ell)} = \mathbb{1}_m \oslash K\mathbf{b}^{(\ell-1)}$, $\mathbf{b}^{(\ell)} = \mathbb{1}_m \oslash K^T\mathbf{a}^{(\ell)}$ where $\oslash$ denotes component-wise division

**Output**: $OT_{\epsilon} (\widetilde{\mathbf{X}},\mathbf{X}) = \sum_{i,j}C_{ij}a_{i}^{(L)}K_{ij}b_{j}^{(L)}$

### Gromov-Wasserstein Distance $GW (\mu_{\theta},\xi)$

Typically, for cross domain learning, we map inputs from both source and target domains into a common embedding space and evaluate their matching in this common space. The Gromov-Wasserstein distance seeks to compare the similarity of the generated target domain distribution $\mu_{\theta}$ and source domain distribution $\xi$.

- Denote the distance function over the *source domain* $\mathcal{S}$ as $d$
  $$
  d:\mathcal{S}\times\mathcal{S}\to\mathbb{R}_+.
  $$

- Denote the distance function over the *target domain* $\mathcal{T}$ as $c$
  $$
  c:\mathcal{T}\times\mathcal{T}\to\mathbb{R}_+.
  $$

- The Gromov-Wasserstein distance is defined as a *relational* distance between the distances on each respective domain.
  $$
  \Pi=\left\{\pi\in\mathbb{R}^{m\times m}_+\mid\forall{i}\sum_{j}\pi_{ij}=1,\forall{j}\sum_{i}\pi_{ij}=1 \right\}\\
  GW(\mu_{\theta},\xi)=\min_{\pi\in\Pi}\sum_{i,j,k,l}\lvert c_R(\tilde{x}_i, \tilde{x}_k) - d(y_j, y_l) \rvert^2 \pi_{ij} \pi_{kl}.
  $$
  Here $\Pi$ defines the set of all joint distributions with marginals $\xi$ and $\mu_{\theta}$. The goal is to find the optimal transport $\pi$ minimizing the distance between intra-space costs $c$, $d$.

- Similar to the Kantorovich problem, entropic regularization is introduced. The above problem may subsequently be solved via projected gradient descent.

### Computing the GW Distance for 2 independent batches of $m$ samples

**Input**: generated sequences $\widetilde{\mathbf{X}}=\{\tilde{x}\}_{i=1}^m,\widetilde{\mathbf{X}}'=\{\tilde{x}'\}_{i=1}^m$

**Input**: source sequences $\mathbf{Y}=\{y\}_{i=1}^m,\mathbf{Y}'=\{y'\}_{i=1}^m$

**Hyperparameters**: regularization $\varepsilon$, projection iterations $M$, Sinkhorn iterations $L$

1. Initialize $\pi^{(0)}_{ij} = \frac{1}{n} \forall{i,j}$

2. Compute Cost Matrices $C_{ij}=c_R(\tilde{x}_i,\tilde{x}'_j)$ and $D_{ij} = d(y_i,y'_j)$

3. For $l = 1:M$

   ​	$E = \frac{1}{m}D^2 \mathbb{1}_m \mathbb{1}_m^T + \frac{1}{m} \mathbb{1}_m \mathbb{1}_m^T C^2 - 2D \pi^{(l-1)} C^T$

   ​	$K = \exp(-E/\varepsilon)$

   ​	$\mathbf{b}^{(0)}  = \mathbb{1}_m	$

   ​	For $\ell = 1:L$

   ​		$\mathbf{a}^{(\ell)} = \mathbb{1}_m \oslash K\mathbf{b}^{(\ell-1)}$, $\mathbf{b}^{(\ell)} = \mathbb{1}_m \oslash K^T\mathbf{a}^{(\ell)}$

   ​	$\pi^{(l)}=\text{diag}(\mathbf{a}^{(L)})K\text{diag}(\mathbf{b}^{(L)})$

**Output**: $GT_{\varepsilon}(\widetilde{\mathbf{X}},\widetilde{\mathbf{X}}',\mathbf{Y},\mathbf{Y}') = \displaystyle \sum_{i,j,k,l}\lvert D_{ik}-C_{jl}\rvert^2\pi_{ij}^{(M)}\pi_{kl}^{(M)}$

## Overall Algorithmic Pipeline

**Input**: source domain with data distribution $\xi$

**Input**: target domain with data distribution $\nu$

**Hyperparameters**: regularization parameters $\epsilon, \varepsilon$, batch size $m$, learning rate $\alpha$, training epochs $T$

1. Initialize generator network parameters $\theta_0$

2. For $l = 1:T$

   ​	Sample two independent mini-batches of source-target sequences $(\mathbf{X},\mathbf{Y}),(\mathbf{X}',\mathbf{Y}')$

   ​	Generate sequences as $\widetilde{\mathbf{X}}=g_{\theta}(\mathbf{Y})$ and $\widetilde{\mathbf{X}}'=g_{\theta}(\mathbf{Y}')$

   ​	Compute $OT_{\epsilon} (\widetilde{\mathbf{X}},\mathbf{X})$ with above algorithm

   ​	The unbiased OT distance is given by 
   $$
   \overline{OT}_{\epsilon}=OT_{\epsilon}(\widetilde{\mathbf{X}},\mathbf{X})+OT_{\epsilon}(\widetilde{\mathbf{X}}',\mathbf{X})+OT_{\epsilon}(\widetilde{\mathbf{X}},\mathbf{X}')
   +OT_{\epsilon}(\widetilde{\mathbf{X}}',\mathbf{X}')-2OT_{\epsilon}(\widetilde{\mathbf{X}},\widetilde{\mathbf{X}}')-2OT_{\epsilon}(\mathbf{X},\mathbf{X}')​
   $$
   ​	Compute $GT_{\varepsilon}(\widetilde{\mathbf{X}},\widetilde{\mathbf{X}}',\mathbf{Y},\mathbf{Y}')$ with above algorithm

   ​	$\theta \leftarrow \theta-\alpha\nabla_{\theta}\overline{OT}_{\epsilon}-\alpha\nabla_{\theta}GW_{\varepsilon}$

**Output**: Generator network parameters $\theta$
