---
layout: default
title: "History of the Photon Vortex Idea"
---

# History of the Photon Vortex Idea

The idea that particles might be knotted vortices of light is older than quantum mechanics itself. What follows is a history of the researchers who pursued this idea, from Victorian Scotland to the present day, and how their work connects to the Null Worldtube Theory.

## 1. Lord Kelvin and the Vortex Atom (1867)

In 1867, Sir William Thomson (Lord Kelvin) proposed that atoms were knotted vortex rings in a frictionless fluid ether. Different elements would correspond to different knot types &mdash; the periodic table as a table of knots. The idea was inspired by Hermann von Helmholtz's proof that vortex rings in an ideal fluid are indestructible: they cannot be created or destroyed, only transformed. This sounded a great deal like the conservation of matter.

The program attracted serious attention for two decades. Peter Guthrie Tait began systematically tabulating knots (the origin of mathematical knot theory), and J.J. Thomson &mdash; who would later discover the electron &mdash; worked on vortex stability. But by the 1890s the vortex atom was abandoned. The luminiferous ether had no experimental support, the program could not explain chemical bonding or spectra, and the discovery of the electron in 1897 redirected physics toward point particles.

The topology was right. The physics was missing.

**References:**
- Thomson, W. "On Vortex Atoms." *Proceedings of the Royal Society of Edinburgh* 6, 94-105 (1867). Reprinted in *Philosophical Magazine* Series 4, 34, 15-24 (1867).
- Helmholtz, H. "&Uuml;ber Integrale der hydrodynamischen Gleichungen, welche den Wirbelbewegungen entsprechen." *Journal f&uuml;r die reine und angewandte Mathematik* 55, 25-55 (1858). English translation by P.G. Tait: *Philosophical Magazine* Series 4, 33(226), 485-512 (1867).
- Tait, P.G. "On Knots." *Transactions of the Royal Society of Edinburgh* 28, 145-190 (1877).
- Thomson, J.J. *A Treatise on the Motion of Vortex Rings.* London: Macmillan (1883).

## 2. F. Ray Skilton and the Pythagorean Triple (1986&ndash;1988)

A century after Kelvin, **F. Raymond Skilton** &mdash; a professor of Computer Science and Information Processing at Brock University in St. Catharines, Ontario &mdash; published three papers in the proceedings of the Annual Pittsburgh Conference on Modeling and Simulation. Working entirely with integer arithmetic, Skilton proved that the fine-structure constant could be derived from a Pythagorean triple:

> 88<sup>2</sup> + 105<sup>2</sup> = 137<sup>2</sup>

giving 1/&alpha; = &radic;(137<sup>2</sup> + &pi;<sup>2</sup>) = 137.036016, matching the measured value to 0.12 parts per million with zero free parameters.

The papers received zero citations. They were published in minor conference proceedings, never digitized, and never indexed by any search engine. Skilton passed away in 1993, at approximately 57 years of age &mdash; five years after his last paper and before the world wide web could have preserved his work.

In February 2026, physical copies were located at the University of Washington Engineering Library, in conference proceedings volumes with crumbling bindings. NWT reveals why Skilton's formula works: the generators of his Pythagorean triple are the torus quantum numbers (p, q, k) = (2, 1, 3). The forgotten triple is the geometric signature of the electron.

Skilton's papers &mdash; possibly the only surviving copies &mdash; have been scanned and are [available in the GitHub repo](https://github.com/JimGalasyn/null-worldtube/tree/main/papers/Skilton).

**References:**
- Skilton, F.R. "Foundation for an integer-based cosmological model." *Proc. 17th Annual Pittsburgh Conf. on Modeling and Simulation*, Vol. 17, Part 1 (1986), pp. 295-300.
- Skilton, F.R. "Foundation for an integer-based cosmological model &mdash; Part 2: Evenness." *Proc. 18th Annual Pittsburgh Conf.*, Vol. 18, Part 5 (1987), pp. 1623-1630.
- Skilton, F.R. "Foundation for an integer-based cosmological model &mdash; Part 3: Integers and the Natural Constants." *Proc. 19th Annual Pittsburgh Conf.*, Vol. 19, Part 1 (1988), pp. 9-12.

## 3. Williamson and van der Mark: Is the Electron a Photon? (1997&ndash;2021)

In 1997, **John G. Williamson** and **Martin B. van der Mark** asked the question directly: *"Is the electron a photon with toroidal topology?"* Their paper, presented at a conference in memory of Louis de Broglie, proposed that the electron is a single wavelength of light twisted into a toroidal configuration, with charge and mass emerging from the topology of the trapped field.

Williamson, an electrical engineer at the University of Glasgow and later founder of the Quantum Bicycle Society in Scotland, developed this into a full algebraic program using Clifford algebra Cl<sub>1,3</sub>. His key contributions:

- **The pivot field** (2008): A Lorentz-invariant scalar P &mdash; the divergence of the four-potential, usually set to zero by the Lorenz gauge condition. Williamson proposed keeping P nonzero as a physical quantity. The cross-term PE in the energy-momentum density creates an inward-directed confining force for radial electric fields, providing a mechanism for trapping electromagnetic energy.
- **Extended Maxwell equations**: In Clifford algebra, Maxwell's equations generalize to a 16-component equation where mass, spin, and electromagnetic field emerge on the same footing &mdash; not added by hand.
- **The Mexican hat potential** (2021): With van der Mark, Williamson showed that a Higgs-like quartic potential arises naturally from the structure of the Clifford algebra Cl(1,3). Spontaneous symmetry breaking &mdash; and therefore mass &mdash; is algebraic, not postulated.

NWT tested Williamson's pivot field in FDTD simulation and found it insufficient for soliton stabilization: the confinement force is linear in the fields and cannot overcome dispersion. The Euler-Heisenberg QED vacuum polarization, with its F<sup>4</sup> nonlinearity, provides the self-focusing that the pivot requires. This is not a rejection of Williamson's program but its completion &mdash; NWT replaces his ad hoc confinement mechanism with known QED physics.

**References:**
- Williamson, J.G. and van der Mark, M.B. "Is the electron a photon with toroidal topology?" *Annales de la Fondation Louis de Broglie* 22(2), 133-160 (1997).
- Williamson, J.G. "On the nature of the electron and other particles." Preprint (2008).
- Van der Mark, M.B. and Williamson, J.G. "Relativistic Inversion, Invariance and Inter-Action." *Symmetry* 13, 1117 (2021).
- Butler, P.H., Gresnigt, N.G., van der Mark, M.B. and Renaud, P.F. "A fields only version of the Lorentz Force Law: Particles replaced by their fields." arXiv:1211.6072 (2012).

## 4. Daniele Funaro: The Mathematician's Electron (2009&ndash;2024)

**Daniele Funaro**, a mathematician at the University of Modena and Reggio Emilia, independently arrived at the torus electron through a completely different route: rigorous analysis of Maxwell's equations on toroidal domains. His work represents the most complete theoretical predecessor to NWT.

Funaro's key results:

- **The no-go theorem**: Standard Maxwell equations admit no soliton solutions. Any bounded, finite-energy EM field in free space must disperse. This is provable via Liouville's theorem: such fields would have to be holomorphic and bounded, hence constant, hence zero. If particles are trapped light, something beyond standard Maxwell is required.
- **Modified Maxwell with a velocity field**: Funaro introduced a velocity field V with magnitude c that enlarges the solution space to include solitons with compact support, while preserving the same Lagrangian.
- **Bessel modes on the torus**: Rotating wave solutions in the torus cross-section, with integer mode number k &ge; 2 &mdash; modes with k = 1 cannot satisfy continuity at the axis. This selection rule independently matches NWT's finding that only certain torus knot modes form stable solitons.
- **The vortex-ring electron** (2009): A cylinder of rotating electromagnetic fields bent into a torus, with explicit formulas for charge, mass, and magnetic moment. The mass turns out to be independent of frequency and tube radius &mdash; it depends only on the Bessel mode number k and the major radius. Mass from topology, not from energy.
- **The proton as a Hill's vortex** (2012): An outer spherical vortex encapsulating an inner toroidal ring, with three charge regions (negative-positive-negative) resembling quark structure. Funaro noted that the proton is not a sign-flipped electron &mdash; it has fundamentally different topology.

Funaro also proved a formal connection between vortex ring filaments and the nonlinear Schr&ouml;dinger equation, noted the Euler fluid analogy for electromagnetic fields, and extended the framework to include Einstein's field equations. In a related paper with collaborators (Chinosi, Della Croce & Funaro 2010), he solved the eigenvalue problem on toroidal domains using finite elements, finding that curvature produces a potential term proportional to 1/(y+&eta;)<sup>2</sup> and that domain shape must be iteratively optimized for rotational degeneracy. The authors noted explicitly: *"We are not able to discuss the stability"* &mdash; they needed nonlinear dynamics.

His work earned a **Gravity Research Foundation Honorable Mention** in 2023 for a paper arguing that spacetime deformations of electromagnetic origin are far from negligible.

NWT completes Funaro's program. Where he had the geometry and the linear mode analysis, NWT adds the Euler-Heisenberg nonlinearity for soliton stability, KAM theory for mode selection, and full FDTD time-domain simulation for dynamical verification. If Funaro was Kepler &mdash; the right geometric picture, calling out for the dynamical theory &mdash; NWT aspires to be Newton.

**References:**
- Funaro, D. "Electromagnetic Radiations as a Fluid Flow." arXiv:0911.4848 (2009).
- Funaro, D. "A Lagrangian for Electromagnetic Solitary Waves in Vacuum." arXiv:1008.2103 (2010).
- Chinosi, C., Della Croce, L. and Funaro, D. "Rotating Electromagnetic Waves in Toroid-Shaped Regions." *Int. J. Modern Physics C* 21(1), 11-32 (2010).
- Funaro, D. *From Photons to Atoms: The Electromagnetic Nature of Matter.* arXiv:1206.3110 (2012); revised edition: World Scientific (2019).
- Funaro, D. "Trapping Electromagnetic Solitons in Cylinders." arXiv:1304.2287 (2013).
- Funaro, D. *Light and Matter, Two Sides of the Same Coin.* Cambridge Scholars (2024).

## 5. Parallel Threads

The photon vortex idea was discovered independently by several researchers working in isolation from one another. None of them knew about Skilton or, in most cases, about each other.

### Vivian Robinson: Subharmonic Frequencies (2011)

**Vivian Robinson**, a New Zealand independent researcher, proposed in his self-published book *The Common Sense Universe* that particles are spinning photons whose masses correspond to subharmonic frequency ratios: f/3, f/9, f/27, and so on. While the book was not peer-reviewed, the subharmonic insight was prescient. NWT identifies these ratios as arising from the mode structure of torus knots: each subharmonic corresponds to a specific winding number on the torus, with the mass hierarchy set by phase closure and confinement. Robinson had the right pattern; torus geometry explains why it appears.

- Robinson, V. *The Common Sense Universe.* Self-published (2023). ISBN 978-0-6454125-5-0.

### Shixing Weng: The Helical Photon (2016)

**Shixing Weng**, working in Brampton, Ontario (near Skilton's Brock University, though decades later), solved the four-potential wave equations in cylindrical coordinates and found that a photon is naturally described as a helical electromagnetic wave on a cylinder of radius r<sub>0</sub> = &lambda;/2&pi;. The photon energy E = h&nu; and angular momentum J = &#8463; both follow from surface integrals over this cylinder. The photon is the *open* (propagating) version of the NWT torus; the electron is the *closed* (trapped) version &mdash; the same helix with its ends joined.

- Weng, S. "A Classical Model of the Photon." *Progress in Physics* 12(1), 49-55 (2016).
- Weng, S. "Theoretical Study on Polarized Photon." *Progress in Physics* 17, 23-29 (2021).

### G&uuml;nter Poelz: Synchrotron Self-Consistency at DESY (2013)

**G&uuml;nter Poelz**, a retired experimental physicist from DESY and Hamburg University, calculated the exact Li&eacute;nard-Wiechert retarded fields of a massless charge circulating at c on a circular orbit of radius r<sub>Q</sub> = &#8463;<sub>C</sub> (the reduced Compton wavelength). Summing the synchrotron radiation modes and requiring self-consistency, he recovered the elementary charge e = 1.6 &times; 10<sup>&minus;19</sup> C and the fine-structure constant &alpha; = 1/137. His field-line plots at higher multipole orders trace out trefoil and figure-eight patterns &mdash; torus knots &mdash; though he did not make the knot-theory connection.

- Poelz, G. "On the Wave Character of the Electron." arXiv:1206.0620 (2013).

### Jack Avrin: Torus Knot Particles (2012)

**Jack S. Avrin**, an independent researcher in California, published a 77-page paper in *Symmetry* modeling all particles as (2,n) torus knots, with the unknot and trefoil as fundamental building blocks. Spin-1/2 follows from the M&ouml;bius strip's double cover of SO(3), three generations from icosahedral symmetry, and CPT invariance from the topology of knot inversions. Avrin had the right geometric intuition: particles *are* torus knots, not merely described by them. NWT adds the dynamics and the quantitative mass predictions that Avrin's purely topological framework lacked.

- Avrin, J.S. "Knots on a Torus: A Model of the Elementary Particles." *Symmetry* 4, 39-115 (2012). DOI: 10.3390/sym4010039.

### Yaroslav Klyushin: The Ether Torus (2015)

**Yaroslav Klyushin** of St. Petersburg modeled particles as toroidal vortices in an ether, with explicit calculations of poloidal velocities and ether densities. His electron has a poloidal-to-meridional velocity ratio of 2.5 (close to NWT's aspect ratio of ~2.9), and his vacuum density of ~10<sup>5</sup>&ndash;10<sup>9</sup> kg/m<sup>3</sup> overlaps NWT's &rho;<sub>0</sub> = 1.69 &times; 10<sup>5</sup> kg/m<sup>3</sup>. He recovered the Rydberg formula from 137 quantized vortex rings between proton and electron. The ether framework is unjustified, but the numerical coincidences are striking.

- Klyushin, Y. "Elementary Particles Structures." Ch. 16 of *Electricity, Gravity, Heat: Another Look* (2nd ed.). International Scientists' Club, St. Petersburg (2015).

### Josef Vrba: Maxwellian Solitons (2022)

**Josef Vrba** reformulated Maxwell's equations as three coupled vector-algebraic relations M(u, B, E), where u is the velocity field, and showed that closed-path solutions &mdash; "3D rotons" tracing torus-knot trajectories &mdash; reproduce the properties of massive particles. The frequency ratio &omega;<sub>1</sub>:&omega;<sub>2</sub> determines the knot type, quantization follows from rational winding ratios, and the Planck relation E = hf is derived from geometric periodicity rather than postulated. His fine-structure relation &kappa;<sup>2</sup> = 1/(2&alpha;) emerges geometrically. The parallel to NWT is nearly exact in topology and differs mainly in method: Vrba works analytically, NWT adds FDTD simulation, nonlinear dynamics, and multi-particle systems.

- Vrba, J. "General Maxwellian Dynamics / Particles are Maxwellian Solitons." *Proceedings: Harbingers of Neophysics* (2022). neophysics.org/p/1673.

### Peter Mohr: Maxwell = Dirac (2010)

At the mainstream end of the spectrum, **Peter J. Mohr** of NIST published a rigorous proof in *Annals of Physics* that Maxwell's equations can be recast exactly as a massless spin-1 Dirac equation: &gamma;<sup>&mu;</sup>&part;<sub>&mu;</sub>&Psi; = 0, where &Psi; is a six-component photon wavefunction built from E and B. The Maxwell Green function has the same covariant form as the Dirac Green function. This is not an analogy &mdash; it is a mathematical identity. If the photon wavefunction *is* the electromagnetic field, then a trapped electromagnetic field on a torus *is* a particle wavefunction. Mohr provides the rigorous foundation for treating NWT's FDTD fields as quantum states.

- Mohr, P.J. "Solutions of the Maxwell equations and photon wave functions." *Annals of Physics* 325, 607-663 (2010).

## 6. The Quantum Topology of Torus Knots

A parallel mathematical tradition, largely unaware of the physics program, developed the tools needed to classify and compute invariants of torus knots.

**Marc Rosso and Vaughan Jones** (Fields Medalist, 1990) derived a universal formula for the quantum invariant of any torus knot in any representation of any simple Lie algebra (1993). For SU(2), this gives the Jones polynomial &mdash; a knot invariant that, evaluated at roots of unity, produces discrete spectra from pure topology. The Rosso-Jones formula involves the *Adams operation* (plethysm), a mathematical structure that counts the ways representations decompose under symmetry &mdash; suggestively similar to how NWT's confinement factor n<sub>q</sub><sup>q</sup> counts the compounding of quark interactions over poloidal windings.

**V.V. Sreedhar** (2015), **Praloy Das, Souvik Pramanik and Subir Ghosh** (2016), **Dripto Biswas and Subir Ghosh** (2019), and **Anjali S and Saurabh Gupta** (2022) developed the classical and quantum mechanics of a particle constrained to a torus knot. Key results include: the effective mass depends on the winding ratio q/p; the Schr&ouml;dinger equation reduces to a Mathieu equation (whose eigenvalues have band structure = mode-locking plateaus); curvature and torsion induce geometric potentials; and the torus knot path requires *two* quantization conditions (EBK semiclassical quantization), naturally producing two quantum numbers from a one-dimensional path.

**Alexander Gorsky, Alexei Milekhin and Nikita Sopenko** (2015) showed in *JHEP* that in Omega-deformed 5D supersymmetric QED, the parameters (n, m) of a torus knot T<sub>n,m</sub> correspond directly to instanton charge and electric charge &mdash; *knot quantum numbers are particle quantum numbers*. The HOMFLY polynomial of the torus knot provides an entropic factor counting the degeneracy of instanton-W-boson configurations. Condensate formation arises from knot topology. This is mainstream gauge theory arriving at the same structure as NWT from a completely different direction.

**References:**
- Rosso, M. and Jones, V. "On the Invariants of Torus Knots Derived from Quantum Groups." *J. Knot Theory Ramifications* 2(1), 97-112 (1993). DOI: 10.1142/S0218216593000064.
- Sreedhar, V.V. "The Classical and Quantum Mechanics of a Particle on a Knot." arXiv:1501.01098 (2015).
- Das, P., Pramanik, S. and Ghosh, S. "Particle on a Torus Knot: Constrained Dynamics and Semi-Classical Quantization in a Magnetic Field." arXiv:1511.09035 (2016).
- Biswas, D. and Ghosh, S. "Quantum Mechanics of Particle on a Torus Knot: Curvature and Torsion Effects." arXiv:1908.06423 (2019).
- Anjali, S. and Gupta, S. "Particle on a torus knot: Symplectic analysis." *Eur. Phys. J. Plus* 137, 511 (2022). DOI: 10.1140/epjp/s13360-022-02699-3.
- Gorsky, A., Milekhin, A. and Sopenko, N. "The condensate from torus knots." *JHEP* 09, 102 (2015). arXiv:1506.06695.

## 7. The Nonlinear Vacuum

NWT's soliton stability relies on the Euler-Heisenberg effective Lagrangian &mdash; the one-loop QED correction that makes the vacuum behave as a nonlinear medium at strong field strengths. This is not speculative physics: it was calculated by Werner Heisenberg and Hans Euler in 1936, rederived by Julian Schwinger in 1951 using proper-time methods, and is the same physics responsible for vacuum birefringence (being measured by PVLAS and LUXE experiments) and Delbr&uuml;ck scattering (measured since 1975).

**Gerald Dunne** (2004, 2012) wrote the standard reviews of the Euler-Heisenberg effective action, covering vacuum birefringence, photon splitting, Schwinger pair production, light-by-light scattering, and extensions to inhomogeneous backgrounds and curved spacetime. NWT's use of EH as a soliton-forming nonlinearity is an application of known, measured QED &mdash; not an ad hoc addition.

**Shiva Kumar** (2026), a nonlinear optics specialist at McMaster University, independently arrived at the same philosophical program from a different direction. Published in *Nature Scientific Reports*, Kumar showed that a nonlinear extension of Maxwell's equations with a &kappa;A<sup>2</sup>A<sub>&mu;</sub> self-interaction produces dark soliton solutions with quantized energies, pseudo-charges whose coupling reproduces the fine-structure constant, and an envelope equation that reduces to the Schr&ouml;dinger equation. His cavity modes equilibrate via four-wave mixing to reproduce the blackbody spectrum. Kumar's work validates the broader "quantum mechanics from nonlinear electrodynamics" paradigm. NWT adds what his framework lacks: the torus topology that gives discrete quantum numbers, mode selection, and the mass spectrum.

**References:**
- Heisenberg, W. and Euler, H. "Folgerungen aus der Diracschen Theorie des Positrons." *Zeitschrift f&uuml;r Physik* 98, 714-732 (1936). DOI: 10.1007/BF01343663.
- Schwinger, J. "On Gauge Invariance and Vacuum Polarization." *Physical Review* 82(5), 664-679 (1951). DOI: 10.1103/PhysRev.82.664.
- Dunne, G.V. "Heisenberg-Euler Effective Lagrangians: Basics and Extensions." arXiv:hep-th/0406216 (2004). In *From Fields to Strings: Circumnavigating Theoretical Physics* (Shifman, Vainshtein, Wheater, eds.).
- Dunne, G.V. "The Heisenberg-Euler Effective Action: 75 years on." *Int. J. Mod. Phys. A* 27, 1260004 (2012). arXiv:1202.1557.
- Kumar, S. "The connection between nonlinear extension of Maxwell's equations, blackbody spectrum, Lorentz force and quantum mechanics." *Scientific Reports* 16, 4269 (2026). DOI: 10.1038/s41598-025-34478-2.

## 8. Torus Knots in the Laboratory

In 2024, **Maitreyi Jayaseelan** and colleagues at the University of Rochester created torus knot wavefunctions experimentally in spinor Bose-Einstein condensates and published the results in *Nature Communications Physics*. By coordinating orbital and spin rotations of atomic wavefunctions, they engineered (p,q) torus knots, M&ouml;bius strips, and Solomon's knots in a multi-component superfluid described by the Gross-Pitaevskii equation &mdash; the same equation NWT uses for the superfluid vacuum.

Their knots are externally engineered by laser fields; NWT claims self-consistent solitons that form spontaneously. But the mathematical framework &mdash; GPE + torus topology &mdash; is identical. Their experimental platform could, in principle, test NWT predictions by measuring the eigenfrequency spectrum of specific torus knot modes and comparing to the phase closure calculation.

**References:**
- Jayaseelan, M., Murphree, J.D., Schultz, J.T., Ruostekoski, J. and Bigelow, N.P. "Topological atom optics and beyond with knotted quantum wavefunctions." *Communications Physics* (Nature) 7, 7 (2024). DOI: 10.1038/s42005-023-01499-0.

## 9. The Null Worldtube

NWT sits at the confluence of these streams. From Kelvin it inherits the vortex topology. From Skilton, the Pythagorean integers. From Williamson, the question: *is the electron a photon?* From Funaro, the Bessel modes and the torus eigenvalue problem. From Robinson, the subharmonic pattern. From Vrba, the algebraic soliton picture. From the quantum topologists, the knot invariants. From Euler-Heisenberg, the nonlinear vacuum that makes it all work.

What NWT adds is the quantitative programme: 56 particle masses to 0.40% median error from four quantum numbers and no free parameters; all three gauge coupling constants from the crossing geometry of the torus; both quark and neutrino mixing matrices from the mod-2 and mod-3 symmetries of the Hopf link and trefoil; CP violation from the geometric phase at crossings; the Higgs mechanism from the Gross&ndash;Pitaevskii nonlinearity at Hopf link crossings; and &mdash; most fundamentally &mdash; the Standard Model gauge group SU(3) &times; SU(2) &times; U(1) itself, derived as the crossing exchange algebra of the trefoil (3 self-crossings &rarr; su(3)) and the Hopf link (2 inter-crossings &rarr; su(2) &oplus; u(1)).

The photon vortex idea was discovered and rediscovered at least a dozen times over 150 years, by physicists and mathematicians and computer scientists working in complete isolation. Each found a piece of the picture. NWT attempts to assemble them &mdash; and, with the gauge group derivation, to show that the pieces fit together into a single mathematical structure: the topology of the simplest knot and the simplest link.

---

[Home](index.html) &#183; [The Predictions](results.html) &#183; [Papers](papers.html) &#183; [Concordance](concordance.html) &#183; [About](about.html)
