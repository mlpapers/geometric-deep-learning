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

- **[Geometric blueprint](https://arxiv.org/abs/2104.13478)** (2021 review). Organizes architectures by domain geometry, symmetry and scale: grids, groups, graphs and manifolds. Invariant outputs stay fixed under a group action; equivariant outputs transform predictably with the input.

- **[When symmetry improves generalization](https://arxiv.org/abs/2102.10333)** (2021). Elesedy and Zaidi prove a benefit for linear models when the target and distribution obey the specified compact-group symmetry. This supports matched structural priors, not the claim that every imposed symmetry improves every task.

- **[Scalar invariant theory](https://arxiv.org/abs/2106.06610)** (2021). Uses invariant scalar coefficients and geometric inputs to parameterize equivariant vector functions. SO(d) and O(d) differ on reflections; this does not establish universality of local scalar message passing.

- **[Permutation symmetry versus graph expressivity](https://arxiv.org/abs/1810.00826)** (2018). GIN matches the 1-WL test under stated aggregation assumptions. Permutation invariance is necessary for graph-level predictions but does not ensure that different graph structures can be distinguished.

- **[Geometric Weisfeiler–Leman](https://proceedings.mlr.press/v202/joshi23a.html)** (2023). Tests discrimination of geometric graphs modulo symmetry. Equivariant propagation can distinguish arrangements lost by locally invariant aggregation; conclusions depend on retained geometry, tensor order and graph connectivity.

- **[Invariant completeness](https://arxiv.org/abs/2402.04836)** (2024). GeoNGNN and specified DimeNet, GemNet and SphereNet variants achieve E(3)-completeness with full connectivity and sufficient depth. Equivariant hidden features are not necessary; practical cutoff implementations do not inherit the theorem.

## Sets and point clouds

- **[Deep Sets](https://arxiv.org/abs/1703.06114)** (2017). Applies an elementwise map, sums latent features and maps the result to an output. The construction is permutation invariant; representation theorems require assumptions about the input domain and latent space.

- **[Limits of sum-decomposition](https://arxiv.org/abs/1901.09006)** (2019). For continuous representations of scalar sets, universal sum-decomposition needs latent dimension at least the maximum set size. Invariance alone does not make a fixed-width pooled representation universally expressive.

- **[Set Transformer](https://arxiv.org/abs/1810.00825)** (2018). Uses attention to model interactions among set elements and invariant pooling for set outputs. Inducing points reduce attention cost to linear in set size when their number is fixed.

- **[PointNet](https://arxiv.org/abs/1612.00593)** (2016). Combines shared pointwise features with symmetric max pooling. Handles unordered point clouds directly; permutation invariance does not imply exact rotation invariance, and local geometric interactions are limited.

- **[PointNet++](https://arxiv.org/abs/1706.02413)** (2017). Applies PointNet hierarchically to metric neighborhoods and aggregates multiple scales. Adds local geometric structure and addresses varying sampling density, extending the global pooling of PointNet.

- **[Point Transformer](https://arxiv.org/abs/2012.09164)** (2020). Uses local vector attention with relative position encoding for point-cloud classification and segmentation. Position-aware attention models local geometry; it does not by itself enforce full rotation equivariance.

## Spectral filtering and deformation stability

- **[Spectral graph filters](https://arxiv.org/abs/1312.6203)** (2013). Learn filters in the graph Laplacian's eigenbasis. [ChebNet](https://arxiv.org/abs/1606.09375) (2016) uses polynomial filters with local graph support, avoiding eigendecomposition during filtering. Polynomial degree controls the receptive field.

- **[Wavelet scattering](https://arxiv.org/abs/1101.2286)** (2011). Cascades wavelets, modulus and averaging. Provides deformation-stability bounds under stated regularity assumptions; exact translation invariance is an infinite-window limit. Stability to deformations is distinct from exact symmetry.

## Group convolutions and steerable features

- **[Group-equivariant CNNs](https://arxiv.org/abs/1602.07576)** (2016). Extends convolution and weight sharing to group actions, including discrete rotations and reflections. Equivariance is built into the layer; augmentation instead asks training to learn behavior across transformed examples.

- **[E(2)-steerable CNNs](https://arxiv.org/abs/1911.08251)** (2019). Solves kernel constraints for planar rotations, reflections and their subgroups using group representations. Feature types specify how channels transform; scalar-only channels are one choice, not the general construction.

- **[3D steerable CNNs](https://arxiv.org/abs/1807.02547)** (2018). Builds kernels from an analytic steerable basis to map scalar, vector and tensor fields equivariantly under rigid motions. Continuous equivariance and errors introduced by voxel sampling should be evaluated separately.

- **[Spherical CNNs](https://arxiv.org/abs/1801.10130)** (2018). Defines rotation-equivariant correlation on spherical signals and computes it with generalized Fourier transforms. Avoids the varying distortions produced by flattening a sphere into a planar image.

- **[Tensor field networks](https://arxiv.org/abs/1802.08219)** (2018). Uses spherical-harmonic filters to propagate scalar, vector and higher-order tensor features on 3D points. Equivariance specifies their transformation under rotations, translations and point permutations.

- **[SE(3)-Transformer](https://arxiv.org/abs/2006.10503)** (2020). Combines invariant attention weights with equivariant value features. Models interactions on 3D graphs and point clouds while preserving roto-translation laws; attention and equivariance play different roles.

- **[E(n)-equivariant GNNs](https://arxiv.org/abs/2102.09844)** (2021). Uses invariant distances and coordinate-difference updates without higher-order tensor features. Enforces translations, rotations and reflections; reflection-invariant scalar outputs cannot distinguish mirror-related inputs.

## Surfaces, meshes and local coordinate frames

- **[Geodesic CNNs](https://arxiv.org/abs/1501.06297)** (2015). Constructs local geodesic polar patches on surfaces and learns filters for shape descriptors. Intrinsic neighborhoods support shape correspondence; orientation choices must be handled when defining directional filters.

- **[MoNet](https://arxiv.org/abs/1611.08402)** (2016). Uses learned weighting functions over local pseudo-coordinates to generalize convolution on graphs and manifolds. Coordinates encode domain geometry; several earlier geometric convolutions arise as special cases.

- **[MeshCNN](https://arxiv.org/abs/1809.05910)** (2018). Convolves on mesh edges and pools through learned edge collapses. Uses mesh connectivity and geometric edge features; pooling changes the discretization according to the task rather than using fixed grid subsampling.

- **[DiffusionNet](https://arxiv.org/abs/2012.00888)** (2020). Learns diffusion times for surface communication, with pointwise networks and gradient features. Laplacian-based diffusion connects it to spectral filtering. Transfer across meshes and point samples does not imply invariance to arbitrary shape changes.

- **[Gauge-equivariant CNNs](https://arxiv.org/abs/1902.04615)** (2019). Makes feature transformations consistent under changes of local coordinate frame on a manifold. The icosahedral implementation applies this to spherical signals; gauge changes are coordinate choices, not physical deformations.

- **[Gauge-equivariant mesh CNNs](https://arxiv.org/abs/2003.05425)** (2020). Uses anisotropic kernels and parallel transport to compare orientation-carrying features across vertices. Extends isotropic graph aggregation with intrinsic directional information while respecting local frame changes.

## Canonicalization, averaging and approximate symmetry

- **[Frame averaging](https://arxiv.org/abs/2110.03336)** (2021). Averages a backbone over a covariantly chosen set of transformations. Can enforce exact symmetry without redesigning each layer; computational cost depends on frame size and continuity is a separate property.

- **[Continuity of frames](https://proceedings.mlr.press/v235/dym24a.html)** (2024). Dym et al. show that common unweighted frame constructions can turn smooth functions into discontinuous equivariant ones. Weighted frames preserve continuity under their conditions, qualifying symmetry guarantees based only on averaging.

- **[Approximate equivariance](https://proceedings.mlr.press/v162/wang22aa.html)** (2022). Relaxes weight sharing and steerable-kernel constraints. Improves on unconstrained and strictly equivariant baselines in smoke-plume and jet-flow forecasts; ocean-current results are similar across models after tuning.

## Latent-space geometry

- **[Hyperbolic neural networks](https://arxiv.org/abs/1805.09112)** (2018). Builds feed-forward and recurrent layers with Möbius operations in the Poincaré ball. Negative curvature supplies a prior for hierarchical representations; this concerns latent-space geometry rather than rotation equivariance of physical inputs.

## Higher-order domains

- **[Message-passing simplicial networks](https://arxiv.org/abs/2103.03212)** (2021). Passes messages among simplices through faces and cofaces. Clique-complex lifting can distinguish graphs that 1-WL cannot; the lift adds structure and cost. Higher expressivity does not establish better accuracy on every graph dataset.

## Geometric generative models

- **[Equivariant molecular diffusion](https://arxiv.org/abs/2203.17003)** (2022). Jointly denoises atom types and 3D coordinates using an equivariant network. Coordinates and noise lie in a zero-center-of-gravity subspace; centering handles translations, while equivariant denoising preserves rotational symmetry.

- **[Riemannian flow matching](https://arxiv.org/abs/2302.03660)** (2023). Trains generative vector fields on manifolds. Closed-form geodesics permit simulation-free training on simple geometries; general geometries use spectral distances and simulated conditional paths. Sampling still integrates the learned flow.

## Molecular and atomistic representations

- **[SchNet](https://arxiv.org/abs/1706.08566)** (2017). Uses continuous distance-based filters for atoms off a regular grid. Predicts invariant energies and derives forces from the energy surface; distance-only local representations constrain the geometric information available.

- **[DimeNet](https://arxiv.org/abs/2003.03123)** (2020). Associates messages with directed edges and combines distances with angles using radial and spherical bases. Angular information distinguishes environments that distance-only neighborhood aggregation can confuse.

- **[PaiNN](https://arxiv.org/abs/2102.03150)** (2021). Propagates scalar and vector features to learn equivariant atomic representations. Supports tensorial property prediction and molecular spectra; vector features retain directional information absent from purely invariant hidden states.

- **[NequIP](https://arxiv.org/abs/2101.03164)** (2021). Uses E(3)-equivariant tensor features to learn interatomic energies and forces from quantum calculations. Its data-efficiency results concern tested systems and reference calculations; symmetry does not remove chemical-domain coverage requirements.

- **[MACE](https://arxiv.org/abs/2206.07697)** (2022). Builds higher-body-order equivariant messages. Increasing interaction order can reduce required message-passing depth; body order, geometric tensor order and number of graph layers describe different forms of model capacity.

- **[Allegro](https://arxiv.org/abs/2204.05249)** (2022). Learns strictly local equivariant many-body representations without message passing between atomic environments. Locality supports parallel simulation, while the cutoff limits which interactions the representation can directly model.

## Comparative evidence and evaluation

- **[Controlled symmetry comparisons](https://arxiv.org/abs/2501.01999)** (2025). Rapidash compares equivariant and less constrained convolutions. Gains depend on task geometry; reference frames improve the tested models. Symmetry breaking through equivariant frames differs from breaking equivariance.

- **[Force error versus molecular simulation](https://arxiv.org/abs/2210.07237)** (2022). Across water, molecules, a peptide and materials, low force error does not reliably predict stable trajectories or accurate observables. Evaluate rollout stability and system-specific statistics alongside held-out force errors.

- **[Smooth potentials: eSEN](https://arxiv.org/abs/2502.12147)** (2025). Ablations find neighbor caps, missing cutoff envelopes and discretized representations can impair energy conservation despite similar test errors. Conservation-tested models show stronger error–property correlations in the studied systems.

- **[Open Catalyst 2020](https://arxiv.org/abs/2010.09990)** (2020). Tests energies, forces and relaxation for adsorbates on catalyst surfaces. Splits separate composition and adsorbate generalization; accuracy on configurations of familiar molecules is a different evaluation target.

- **[Matbench Discovery](https://arxiv.org/abs/2308.14920)** (2023). Evaluates thermodynamic-stability screening from unrelaxed crystal structures. Accurate energy regressors can still produce false positives near the convex-hull boundary; discovery precision and recall measure utility that average energy error misses.

## Software

- **[e3nn](https://docs.e3nn.org/en/stable/)** (documentation). Provides irreducible representations, spherical harmonics and equivariant tensor products for E(3) networks. It supplies geometric building blocks; the complete model’s inputs, nonlinearities and outputs must respect the intended representations.

## Related topics

[Graph neural networks](https://github.com/mlpapers/graph-neural-networks) · [Scientific machine learning](https://github.com/mlpapers/scientific-machine-learning) · [Computer vision](https://github.com/mlpapers/computer-vision) · [Feature extraction](https://github.com/mlpapers/feature-extraction) · [Generalization](https://github.com/mlpapers/generalization)
