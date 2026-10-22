# Geometric Deep Learning

| Design choice | Alternatives covered below |
|---|---|
| Input domain | Sets, graphs, point clouds, surfaces, spheres |
| Symmetry and output | Permutations, rigid motions, local frame changes; invariant scalars or transforming outputs |
| Hidden features | Scalars, vectors, higher-order tensors |
| Interactions | Pooling, attention, local messages, spectral filters, diffusion |
| Scale and sampling | Hierarchical pooling, mesh resolution, discretization transfer |
| Latent geometry | Euclidean or hyperbolic representations |

## Symmetry, geometry and expressive power

- **Geometric blueprint** - Organizes architectures by domain geometry, symmetry and scale: grids, groups, graphs and manifolds. Invariant outputs stay fixed under a group action; equivariant outputs transform predictably with the input. [Bronstein et al. 2021](<https://arxiv.org/abs/2104.13478> "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges; arXiv").
- **When symmetry improves generalization** - Elesedy and Zaidi prove a benefit for linear models when the target and distribution obey the specified compact-group symmetry. This supports matched structural priors, not the claim that every imposed symmetry improves every task. [Elesedy and Zaidi 2021](<https://arxiv.org/abs/2102.10333> "Provably Strict Generalisation Benefit for Equivariant Models; arXiv").
- **Scalar invariant theory** - Uses invariant scalar coefficients and geometric inputs to parameterize equivariant vector functions. SO(d) and O(d) differ on reflections; this does not establish universality of local scalar message passing. [Villar et al. 2021](<https://arxiv.org/abs/2106.06610> "Scalars are universal: Equivariant machine learning, structured like classical physics; NeurIPS").
- **Permutation symmetry versus graph expressivity** - GIN matches the 1-WL test under stated aggregation assumptions. Permutation invariance is necessary for graph-level predictions but does not ensure that different graph structures can be distinguished. [Xu et al. 2018](<https://arxiv.org/abs/1810.00826> "How Powerful are Graph Neural Networks?; arXiv").
- **Geometric Weisfeiler-Leman** - Tests discrimination of geometric graphs modulo symmetry. Equivariant propagation can distinguish arrangements lost by locally invariant aggregation; conclusions depend on retained geometry, tensor order and graph connectivity. [Joshi et al. 2023](<https://proceedings.mlr.press/v202/joshi23a.html> "On the Expressive Power of Geometric Graph Neural Networks; ICML").
- **Invariant completeness** - GeoNGNN and specified DimeNet, GemNet and SphereNet variants achieve E(3)-completeness with full connectivity and sufficient depth. Equivariant hidden features are not necessary; practical cutoff implementations do not inherit the theorem. [Li et al. 2024](<https://arxiv.org/abs/2402.04836> "On the Completeness of Invariant Geometric Deep Learning Models; arXiv").

## Sets and point clouds

- **Deep Sets** - Applies an elementwise map, sums latent features and maps the result to an output. The construction is permutation invariant; representation theorems require assumptions about the input domain and latent space. [Zaheer et al. 2017](<https://arxiv.org/abs/1703.06114> "Deep Sets; NeurIPS").
- **Limits of sum-decomposition** - For continuous representations of scalar sets, universal sum-decomposition needs latent dimension at least the maximum set size. Invariance alone does not make a fixed-width pooled representation universally expressive. [Wagstaff et al. 2019](<https://arxiv.org/abs/1901.09006> "On the Limitations of Representing Functions on Sets; ICML").
- **Set Transformer** - Uses attention to model interactions among set elements and invariant pooling for set outputs. Inducing points reduce attention cost to linear in set size when their number is fixed. [Lee et al. 2019](<https://arxiv.org/abs/1810.00825> "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks; ICML").
- **PointNet** - Combines shared pointwise features with symmetric max pooling. Handles unordered point clouds directly; permutation invariance does not imply exact rotation invariance, and local geometric interactions are limited. [Qi et al. 2017](<https://arxiv.org/abs/1612.00593> "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation; CVPR").
- **PointNet++** - Applies PointNet hierarchically to metric neighborhoods and aggregates multiple scales. Adds local geometric structure and addresses varying sampling density, extending the global pooling of PointNet. [Qi et al. 2017](<https://arxiv.org/abs/1706.02413> "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space; NeurIPS").
- **Point Transformer** - Uses local vector attention with relative position encoding for point-cloud classification and segmentation. Position-aware attention models local geometry; it does not by itself enforce full rotation equivariance. [Zhao et al. 2021](<https://arxiv.org/abs/2012.09164> "Point Transformer; ICCV").

## Spectral filtering and deformation stability

- **Spectral graph filters** - Learn filters in the graph Laplacian's eigenbasis. ChebNet uses polynomial filters with local graph support, avoiding eigendecomposition during filtering. Polynomial degree controls the receptive field. [Bruna et al. 2013](<https://arxiv.org/abs/1312.6203> "Spectral Networks and Locally Connected Networks on Graphs; arXiv"), [Defferrard et al. 2016](<https://arxiv.org/abs/1606.09375> "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering; NeurIPS").
- **Wavelet scattering** - Cascades wavelets, modulus and averaging. Provides deformation-stability bounds under stated regularity assumptions; exact translation invariance is an infinite-window limit. Stability to deformations is distinct from exact symmetry. [Mallat 2011](<https://arxiv.org/abs/1101.2286> "Group Invariant Scattering; arXiv").

## Group convolutions and steerable features

- **Group-equivariant CNNs** - Extends convolution and weight sharing to group actions, including discrete rotations and reflections. Equivariance is built into the layer; augmentation instead asks training to learn behavior across transformed examples. [Cohen and Welling 2016](<https://arxiv.org/abs/1602.07576> "Group Equivariant Convolutional Networks; ICML").
- **E(2)-steerable CNNs** - Solves kernel constraints for planar rotations, reflections and their subgroups using group representations. Feature types specify how channels transform; scalar-only channels are one choice, not the general construction. [Weiler and Cesa 2019](<https://arxiv.org/abs/1911.08251> "General $E(2)$-Equivariant Steerable CNNs; NeurIPS").
- **3D steerable CNNs** - Builds kernels from an analytic steerable basis to map scalar, vector and tensor fields equivariantly under rigid motions. Continuous equivariance and errors introduced by voxel sampling should be evaluated separately. [Weiler et al. 2018](<https://arxiv.org/abs/1807.02547> "3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data; NeurIPS").
- **Spherical CNNs** - Defines rotation-equivariant correlation on spherical signals and computes it with generalized Fourier transforms. Avoids the varying distortions produced by flattening a sphere into a planar image. [Cohen et al. 2018](<https://arxiv.org/abs/1801.10130> "Spherical CNNs; ICLR").
- **Tensor field networks** - Uses spherical-harmonic filters to propagate scalar, vector and higher-order tensor features on 3D points. Equivariance specifies their transformation under rotations, translations and point permutations. [Thomas et al. 2018](<https://arxiv.org/abs/1802.08219> "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds; arXiv").
- **SE(3)-Transformer** - Combines invariant attention weights with equivariant value features. Models interactions on 3D graphs and point clouds while preserving roto-translation laws; attention and equivariance play different roles. [Fuchs et al. 2020](<https://arxiv.org/abs/2006.10503> "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks; NeurIPS").
- **E(n)-equivariant GNNs** - Uses invariant distances and coordinate-difference updates without higher-order tensor features. Enforces translations, rotations and reflections; reflection-invariant scalar outputs cannot distinguish mirror-related inputs. [Satorras et al. 2021](<https://arxiv.org/abs/2102.09844> "E(n) Equivariant Graph Neural Networks; ICML").

## Surfaces, meshes and local coordinate frames

- **Geodesic CNNs** - Constructs local geodesic polar patches on surfaces and learns filters for shape descriptors. Intrinsic neighborhoods support shape correspondence; orientation choices must be handled when defining directional filters. [Masci et al. 2015](<https://arxiv.org/abs/1501.06297> "Geodesic convolutional neural networks on Riemannian manifolds; arXiv").
- **MoNet** - Uses learned weighting functions over local pseudo-coordinates to generalize convolution on graphs and manifolds. Coordinates encode domain geometry; several earlier geometric convolutions arise as special cases. [Monti et al. 2017](<https://arxiv.org/abs/1611.08402> "Geometric deep learning on graphs and manifolds using mixture model CNNs; CVPR").
- **MeshCNN** - Convolves on mesh edges and pools through learned edge collapses. Uses mesh connectivity and geometric edge features; pooling changes the discretization according to the task rather than using fixed grid subsampling. [Hanocka et al. 2019](<https://arxiv.org/abs/1809.05910> "MeshCNN: A Network with an Edge; ACM Trans. Graph.").
- **DiffusionNet** - Learns diffusion times for surface communication, with pointwise networks and gradient features. Laplacian-based diffusion connects it to spectral filtering. Transfer across meshes and point samples does not imply invariance to arbitrary shape changes. [Sharp et al. 2020](<https://arxiv.org/abs/2012.00888> "DiffusionNet: Discretization Agnostic Learning on Surfaces; arXiv").
- **Gauge-equivariant CNNs** - Makes feature transformations consistent under changes of local coordinate frame on a manifold. The icosahedral implementation applies this to spherical signals; gauge changes are coordinate choices, not physical deformations. [Cohen et al. 2019](<https://arxiv.org/abs/1902.04615> "Gauge Equivariant Convolutional Networks and the Icosahedral CNN; ICML").
- **Gauge-equivariant mesh CNNs** - Uses anisotropic kernels and parallel transport to compare orientation-carrying features across vertices. Extends isotropic graph aggregation with intrinsic directional information while respecting local frame changes. [de Haan et al. 2021](<https://arxiv.org/abs/2003.05425> "Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs; ICLR").

## Canonicalization, averaging and approximate symmetry

- **Frame averaging** - Averages a backbone over a covariantly chosen set of transformations. Can enforce exact symmetry without redesigning each layer; computational cost depends on frame size and continuity is a separate property. [Puny et al. 2021](<https://arxiv.org/abs/2110.03336> "Frame Averaging for Invariant and Equivariant Network Design; arXiv").
- **Continuity of frames** - Dym et al. show that common unweighted frame constructions can turn smooth functions into discontinuous equivariant ones. Weighted frames preserve continuity under their conditions, qualifying symmetry guarantees based only on averaging. [Dym et al. 2024](<https://proceedings.mlr.press/v235/dym24a.html> "Equivariant Frames and the Impossibility of Continuous Canonicalization; ICML").
- **Approximate equivariance** - Relaxes weight sharing and steerable-kernel constraints. Improves on unconstrained and strictly equivariant baselines in smoke-plume and jet-flow forecasts; ocean-current results are similar across models after tuning. [Wang et al. 2022](<https://proceedings.mlr.press/v162/wang22aa.html> "Approximately Equivariant Networks for Imperfectly Symmetric Dynamics; ICML").

## Latent-space geometry

- **Hyperbolic neural networks** - Builds feed-forward and recurrent layers with Möbius operations in the Poincaré ball. Negative curvature supplies a prior for hierarchical representations; this concerns latent-space geometry rather than rotation equivariance of physical inputs. [Ganea et al. 2018](<https://arxiv.org/abs/1805.09112> "Hyperbolic Neural Networks; NeurIPS").

## Higher-order domains

- **Message-passing simplicial networks** - Passes messages among simplices through faces and cofaces. Clique-complex lifting can distinguish graphs that 1-WL cannot; the lift adds structure and cost. Higher expressivity does not establish better accuracy on every graph dataset. [Bodnar et al. 2021](<https://arxiv.org/abs/2103.03212> "Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks; ICML").

## Geometric generative models

- **Equivariant molecular diffusion** - Jointly denoises atom types and 3D coordinates using an equivariant network. Coordinates and noise lie in a zero-center-of-gravity subspace; centering handles translations, while equivariant denoising preserves rotational symmetry. [Hoogeboom et al. 2022](<https://arxiv.org/abs/2203.17003> "Equivariant Diffusion for Molecule Generation in 3D; ICML").
- **Riemannian flow matching** - Trains generative vector fields on manifolds. Closed-form geodesics permit simulation-free training on simple geometries; general geometries use spectral distances and simulated conditional paths. Sampling still integrates the learned flow. [Chen and Lipman 2024](<https://arxiv.org/abs/2302.03660> "Flow Matching on General Geometries; ICLR").

## Molecular and atomistic representations

- **SchNet** - Uses continuous distance-based filters for atoms off a regular grid. Predicts invariant energies and derives forces from the energy surface; distance-only local representations constrain the geometric information available. [Schütt et al. 2017](<https://arxiv.org/abs/1706.08566> "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions; NeurIPS").
- **DimeNet** - Associates messages with directed edges and combines distances with angles using radial and spherical bases. Angular information distinguishes environments that distance-only neighborhood aggregation can confuse. [Gasteiger et al. 2020](<https://arxiv.org/abs/2003.03123> "Directional Message Passing for Molecular Graphs; ICLR").
- **PaiNN** - Propagates scalar and vector features to learn equivariant atomic representations. Supports tensorial property prediction and molecular spectra; vector features retain directional information absent from purely invariant hidden states. [Schütt et al. 2021](<https://arxiv.org/abs/2102.03150> "Equivariant message passing for the prediction of tensorial properties and molecular spectra; ICML").
- **NequIP** - Uses E(3)-equivariant tensor features to learn interatomic energies and forces from quantum calculations. Its data-efficiency results concern tested systems and reference calculations; symmetry does not remove chemical-domain coverage requirements. [Batzner et al. 2022](<https://arxiv.org/abs/2101.03164> "E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials; Nat Commun").
- **MACE** - Builds higher-body-order equivariant messages. Increasing interaction order can reduce required message-passing depth; body order, geometric tensor order and number of graph layers describe different forms of model capacity. [Batatia et al. 2022](<https://arxiv.org/abs/2206.07697> "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields; NeurIPS").
- **Allegro** - Learns strictly local equivariant many-body representations without message passing between atomic environments. Locality supports parallel simulation, while the cutoff limits which interactions the representation can directly model. [Musaelian et al. 2022](<https://arxiv.org/abs/2204.05249> "Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics; arXiv").

## Comparative evidence and evaluation

- **Controlled symmetry comparisons** - Rapidash compares equivariant and less constrained convolutions. Gains depend on task geometry; reference frames improve the tested models. Symmetry breaking through equivariant frames differs from breaking equivariance. [Vadgama et al. 2025](<https://arxiv.org/abs/2501.01999> "Probing Equivariance and Symmetry Breaking in Convolutional Networks; arXiv").
- **Force error versus molecular simulation** - Across water, molecules, a peptide and materials, low force error does not reliably predict stable trajectories or accurate observables. Evaluate rollout stability and system-specific statistics alongside held-out force errors. [Fu et al. 2023](<https://arxiv.org/abs/2210.07237> "Forces are not Enough: Benchmark and Critical Evaluation for Machine Learning Force Fields with Molecular Simulations; TMLR").
- **Smooth potentials: eSEN** - Ablations find neighbor caps, missing cutoff envelopes and discretized representations can impair energy conservation despite similar test errors. Conservation-tested models show stronger error-property correlations in the studied systems. [Fu et al. 2025](<https://arxiv.org/abs/2502.12147> "Learning Smooth and Expressive Interatomic Potentials for Physical Property Prediction; arXiv").
- **Open Catalyst 2020** - Tests energies, forces and relaxation for adsorbates on catalyst surfaces. Splits separate composition and adsorbate generalization; accuracy on configurations of familiar molecules is a different evaluation target. [Chanussot et al. 2021](<https://arxiv.org/abs/2010.09990> "The Open Catalyst 2020 (OC20) Dataset and Community Challenges; ACS Catal.").
- **Matbench Discovery** - Evaluates thermodynamic-stability screening from unrelaxed crystal structures. Accurate energy regressors can still produce false positives near the convex-hull boundary; discovery precision and recall measure utility that average energy error misses. [Riebesell et al. 2023](<https://arxiv.org/abs/2308.14920> "Matbench Discovery -- A framework to evaluate machine learning crystal stability predictions; arXiv").

## Software

- **e3nn** - Provides irreducible representations, spherical harmonics and equivariant tensor products for E(3) networks. It supplies geometric building blocks; the complete model's inputs, nonlinearities and outputs must respect the intended representations. [docs](<https://docs.e3nn.org/en/stable/>).

## Related topics

- [Graph neural networks](https://mlpapers.org/graph-neural-networks/)
- [Scientific machine learning](https://mlpapers.org/scientific-machine-learning/)
- [Computer vision](https://mlpapers.org/computer-vision/)
- [Feature extraction](https://mlpapers.org/feature-extraction/)
- [Generalization](https://mlpapers.org/generalization/)
