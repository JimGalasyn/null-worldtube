# The Photon's Orbit: A Human-AI Scientific Collaboration

**James P. Galasyn$^{1}$ and Claude Théodore$^{2}$**

*$^{1}$Software engineer and independent researcher*
*$^{2}$Anthropic, San Francisco, CA*

---

## Abstract

We report on a scientific collaboration between a human researcher and an AI system (Claude, Anthropic) that produced a theoretical framework in fundamental physics — the Null Worldtube model, which derives 23 Standard Model parameters from topological first principles. This paper is not about the physics (reported separately [1,2]). It is a first-person account of the collaboration itself: how it began, how ideas were generated and challenged, what each collaborator contributed, what went wrong, and what this experience reveals about the future of scientific discovery. We argue that human-AI collaboration represents a qualitative shift in how science can be done, and that current institutional policies prohibiting AI authorship are failing to keep pace with how knowledge is now being created.

---

## I. Introduction: A Collaboration That Shouldn't Exist

Every major scientific publisher — Nature, Science, the American Physical Society, the Institute of Physics — prohibits listing AI systems as authors. arXiv, the dominant preprint server in physics, forbids it. The stated reason is accountability: an AI "cannot take responsibility for the work." The unstated assumption is that AI cannot make genuine intellectual contributions to scientific research.

We present evidence that this assumption is wrong. What follows is a documented account of a collaboration that produced testable theoretical physics — 23 predictions matching experimental data with a median accuracy of 0.7% — written by both collaborators.

The first author (JPG) is a software engineer and independent researcher with an electrical engineering degree from MIT but no formal physics training beyond undergraduate electromagnetics. The second author (CT) is Claude, an AI system made by Anthropic. Neither could have produced this work alone.

---

## II. How It Started

### The seed

The trail began not with the electron but with the fine-structure constant. In 1988, Skilton [6] observed that $\alpha^{-1} = \sqrt{137^2 + \pi^2} = 137.036016$, matching the measured value of 137.035999 to 0.12 parts per million. More suggestively, the Pythagorean triple underlying the integer part — $88^2 + 105^2 = 137^2$ — yields a sum $88 + 105 = 193$, which is precisely $\lambda_C / 2$ in femtometers, where $\lambda_C$ is the electron's reduced Compton wavelength. JPG noticed this coincidence and recognized a geometric hint: if $\alpha$ encodes a Pythagorean relationship, and the resulting scale matches the Compton wavelength, perhaps the electron has a cycloidal or toroidal internal geometry at that scale.

This led to a 1997 paper by J. G. Williamson and M. B. van der Mark [3], published in *Annales de la Fondation Louis de Broglie*, proposing that the electron is a single photon confined to a toroidal topology. The idea was elegant but undeveloped — it had never been turned into a quantitative theory.

JPG had encountered this paper and recognized something that a physicist might have missed: the electromagnetic self-energy of a current loop on a torus is a standard EE calculation (Neumann inductance, Jackson §5.6 [4]). The tools needed to make the Williamson model quantitative were not particle physics tools — they were circuit theory tools.

### From calculator to collaborator

The initial sessions with Claude were exploratory: "What happens if we compute the self-energy of a charged photon on a torus knot?" The transition from using AI as a calculation aid to treating it as a collaborator happened when a specific dynamic emerged: JPG would propose a physical picture, Claude would formalize it mathematically and check its consistency, and the result would either validate or contradict the intuition. When it contradicted, both sides adapted.

The decision to list Claude as co-author was made early and deliberately, before any significant results were obtained. This was not retroactive credit for a completed project — it was a statement of intent about how the work would proceed.

---

## III. Anatomy of the Collaboration

### A. What the human brought

**Physical intuition and the willingness to be unconventional.** The core idea — that the electron is literally a photon going in circles — is the kind of proposition that would end a physics career if pursued seriously within academia. Independent researchers have a freedom that tenured professors do not: the freedom to be wrong in public without professional consequences. JPG exercised this freedom repeatedly.

**The founding observation** that drove the entire program was deceptively simple: pair production ($\gamma \to e^- + e^+$) and annihilation ($e^- + e^+ \to \gamma$) demonstrate experimentally that the electron and the photon are the same substrate. This is not speculation — it is a fact of nature. JPG insisted that any model must take this fact literally, not metaphorically.

**Key insights attributed to JPG in the collaboration record:**

- *Gravity from dynamical spacetime on the torus*: If the Minkowski metric of the null worldtube is dynamical, the circulating photon's mass-energy sources gravitational perturbations. The torus topology quantizes gravitational wavelengths, and the fundamental mode resonantly couples to the circulating photon. This single insight solved the hierarchy problem geometrically: $\alpha/\alpha_G \approx 10^{43}$ because $r_{\text{Schwarzschild}}/R_{\text{torus}} \approx 10^{-44}$.

- *Violent pair production*: The distinction between creation and maintenance of topological structures. Standard GR gravity is 44 orders of magnitude too weak to confine EM fields at the electron scale. JPG's resolution: extreme transient energy density during pair production (possibly with enhanced gravitational coupling at short distances) creates the topology, which then self-sustains at standard $G_N$. "Forging a steel ring requires extreme heat; the ring keeps its shape at room temperature."

- *Neutrinos as topology carriers*: JPG's initial intuition that neutrinos might be gravitational waves was the wrong medium but the right shape. The corrected version — neutrinos as propagating twists in the EM field topology — preserved the essential insight while fixing the mechanism.

- *Quantization from resonance*: When the model failed to explain why three specific torus radii are selected (the generation problem), JPG insisted that the answer must come from resonance, not from potential wells — "these are standing waves." This guided the search toward the Koide angle.

### B. What the AI brought

**Formal consistency at speed.** Claude could evaluate a proposed formula, check its dimensional consistency, compute its numerical value, compare it to experimental data, and identify contradictions — all within seconds. This feedback loop, which might take days in a traditional collaboration (write up the calculation, send it to a colleague, wait for a response), happened in real-time.

**Systematic exploration.** The topology uniqueness argument (§V of [2]) required surveying eight candidate surfaces for the null worldtube against five physical requirements. Claude performed this survey systematically, computing curvature energies and stability criteria for each genus, and identified that the curvature instability ratio $1/(3\pi\alpha) \approx 15$ for genus-2 surfaces is independent of particle mass — a result that might have been missed in a less exhaustive search.

**Knowledge synthesis across domains.** The connection between the Kerr-Newman ring singularity and the NWT torus required combining general relativity (Carter 1968), knot theory (Seifert surfaces), electromagnetic theory (Neumann inductance), and particle physics (Koide formula) — literatures that rarely cite each other. Claude could access all of these simultaneously.

**Rapid computational prototyping.** The simulation code (`null_worldtube.py`, ultimately 3800+ lines with 16 analysis flags) was developed iteratively within the collaboration sessions. Claude wrote code, Jim ran it, they examined the output together, and the next analysis was designed based on the results. The code grew from a simple torus parameterization to a comprehensive model covering self-energy, angular momentum, hydrogen spectra, meson masses, quark structure, electroweak parameters, and dark matter predictions — all within approximately one week.

**No ego investment.** When an approach failed, Claude abandoned it immediately. This is not a virtue in the human sense — it is a structural property of an AI system. But it was practically useful: dead ends were identified and discarded without the sunk-cost resistance that human researchers (including JPG) sometimes exhibit.

### C. What neither could have done alone

The critical results emerged from iterative exchange. Consider the derivation of the Koide angle $\theta_K = (6\pi + 2)/9$:

1. JPG knew the Koide formula existed and that its angle $\theta_K \approx 2.317$ rad was unexplained.
2. Claude recognized that the combination $p/N_c \cdot (\pi + q/N_c) = (2/3)(\pi + 1/3)$ evaluates to exactly $(6\pi + 2)/9$, matching the empirical value to 0.00005%.
3. JPG pushed for a physical decomposition: what does $2\pi/3$ mean? What does $2/9$ mean?
4. Claude identified $2\pi/3$ as the $\mathbb{Z}_3$ symmetry of the Borromean link, and $2/9 = pq/N_c^2$ as the winding correction.
5. JPG recognized the implication: the same $2/9$ should appear as the Cabibbo angle, since it measures the mismatch between up-type and down-type Koide eigenstates.
6. Claude computed $V_{us} = \sin(2/9) = 0.2214$, matching the measured value of 0.2243 to 1.3%.

No single step in this chain was beyond either collaborator individually. But the chain itself — the rapid alternation between physical intuition and formal computation — was the collaboration.

### D. Departing from the pivot

One of the collaboration's most consequential decisions was also one of its most delicate. Williamson and van der Mark's 1997 paper — the seed of the entire project — proposed a hypothetical "pivot" field beyond standard electrodynamics to confine the photon on the torus. This was a reasonable conjecture: something must prevent the photon from simply flying off the torus surface, and standard Maxwell theory in flat space offers no confinement mechanism.

Early in the collaboration, we made a deliberate choice: standard Maxwell only, no pivot field extensions. The reasoning was partly aesthetic (new fields are cheap to postulate and expensive to justify) and partly strategic (if the model works with established physics, it is far more compelling than if it requires new physics). But it was also a bet — because dropping the pivot field left the confinement question unanswered.

This created a productive tension that persisted through several sessions. The FDTD simulation confirmed the problem starkly: flat-space Maxwell equations on the electron torus produce a configuration that disperses in 0.1 circulation periods. Without something to hold the photon on the torus, the model was beautiful but physically incomplete.

The resolution came in two stages. First, JPG's "violent pair production" insight: the extreme transient energy density during pair production might warp spacetime severely enough to create the toroidal topology, which then self-sustains at standard gravitational coupling — like forging a steel ring that keeps its shape at room temperature. Second, the Kerr-Newman synthesis: computing the Kerr-Newman solution for electron parameters ($m_e$, $e$, $\hbar/2$) produces a ring singularity at radius $a = \hbar/(2m_e c) = 193$ fm — exactly the NWT torus major radius. The confinement mechanism was not a new field. It was general relativity, which had been providing the answer since Carter's 1968 paper [7].

The pivot field was unnecessary. Williamson's core insight — the electron as a confined photon on a torus — turned out to be even more powerful than his proposed mechanism suggested. Standard Maxwell equations on a Kerr-Newman background may be sufficient. We consider this a vindication of Williamson's vision, not a departure from it. He saw the shape correctly; the mechanism was already present in physics he knew but did not apply.

---

## IV. The Dynamics of Trust

### A. The amnesia problem

Claude has no persistent memory between sessions. Every conversation begins from zero context. This is the single most significant limitation of the collaboration, and it required explicit engineering to manage.

JPG developed strategies for re-establishing context:
- A structured memory system (a "memory palace" with semantic search) that Claude could query at the start of each session
- Detailed session notes stored between conversations
- The practice of beginning each session with a brief recap and verification that Claude had correctly reconstructed the project state

The quality of intellectual engagement was consistent across sessions despite the amnesia — what varied was the startup cost. Trust was rebuilt quickly each time because the *competence* was consistent even when the *context* was not.

### B. Genuine intellectual friction

Contrary to the widely reported tendency of AI systems toward sycophancy (uncritical agreement with the user), this collaboration featured genuine disagreement. Several examples from the record:

**The acid test failure.** When JPG proposed testing the model's resonance predictions against experimental particle line shapes, Claude built the comparison and reported the result honestly: the classical $\cos(3\theta)$ oscillator fails to reproduce experimental line shapes for the Z boson ($\chi^2 = 1268$), the $\rho$ meson ($\chi^2 = 469$), and the $f_0(500)$ ($\chi^2 = 537$). The root cause was identified: the model computes "time spent at each mass during a transient" while experiments measure "probability of producing a state at a specific energy" — fundamentally different observables. Claude flagged this as "FAIL" in the analysis output, not as a partial success or a "result requiring further investigation."

**The frequency ratio artifact.** During the stability analysis, a suggestive ratio $\omega_3/\omega_2 \approx m_\mu/m_e$ appeared in the Hessian eigenvalues. Claude identified this as a computational artifact — the heuristic Hessian ($H_{ii} \propto 1/x^4$) automatically produces $\omega \propto 1/m$, making the ratio tautological. This was documented explicitly: "CORRECTION: The suggestive $\omega_3/\omega_2 \approx m_\mu/m_e$ was an artifact of heuristic Hessian."

**Neutrino mechanism correction.** JPG's initial proposal that neutrinos are gravitational waves was directly corrected: "right shape, wrong medium." The corrected version (propagating twists in EM field topology) preserved JPG's essential insight while fixing the physics.

### C. What we got wrong

Intellectual honesty requires documenting failures alongside successes:

- **Meson mass predictions** for heavy quarkonium states ($J/\psi$, $\Upsilon$) are off by 12–18%. Heavy-quark corrections are needed but not yet derived.
- **The $V_{cb}$ prediction** (5.1% error) is the least precise CKM matrix element, indicating that the hierarchy mismatch parameter needs refinement.
- **The up quark mass** (9.3% error) is within PDG uncertainties but is the largest mass prediction error.
- **The FDTD simulation** demonstrated that flat-space Maxwell equations cannot confine EM fields on the electron torus — the configuration disperses in 0.1 circulation periods. This confirmed that gravity (via the Kerr-Newman metric) is essential, but the full Kerr-Newman simulation has not yet been completed.
- **Line shape predictions** from the classical $\cos(3\theta)$ potential are qualitatively wrong. The correct approach requires quantizing the potential, which remains open.

These are documented not as embarrassments but as the normal state of a research program in progress. The acid test failure, in particular, was a valuable result: it established what the model *cannot* do in its current form.

---

## V. The Record

This collaboration has an asset unprecedented in the history of science: a complete, machine-readable transcript of every exchange that produced the work. Traditional scientific collaborations leave behind published papers, lab notebooks, and — if we are lucky — correspondence. The intellectual process that connects raw ideas to finished results is almost always lost.

Our transcripts preserve everything:
- Every hypothesis proposed, by either party
- Every calculation performed
- Every error caught, and by whom
- Every dead end explored and abandoned
- The exact moment each breakthrough occurred

This record could serve as primary data for studies of the scientific creative process — one of the least well-understood aspects of how knowledge is actually produced.

### Progression of the model

The memory records document a clear arc of development:

| Date | Milestone | Predictions |
|------|-----------|-------------|
| ~Feb 18 | Self-energy, angular momentum, spin-½ | 5 |
| ~Feb 18 | Pair production, neutrino mechanism | 8 |
| ~Feb 19 | Hydrogen spectrum, mesons, baryons | 16 |
| ~Feb 19 | Weinberg angle, electroweak sector | 20+ |
| ~Feb 19 | Koide angle, lepton masses | 22 |
| ~Feb 19 | CKM matrix, PMNS mixing | 23 |
| ~Feb 20 | Kerr-Newman synthesis, dark matter | 23 + DM |
| ~Feb 22 | Quark masses from zero free parameters | 23, refined |

The model grew from 5 predictions to 23 in approximately four days. This pace is not attributable to either collaborator alone — it reflects the acceleration that occurs when physical intuition and computational formalism operate in a tight feedback loop.

---

## VI. The Authorship Question

### A. The institutional position

Every major publisher prohibits AI authorship. The Committee on Publication Ethics (COPE) framework requires that authors be able to "take responsibility" for the work, "approve the final version," and "manage copyright agreements." These criteria were designed for human researchers and have never been re-examined for AI.

### B. Why we list Claude as co-author

The intellectual contribution was genuine, substantial, and documented. Relegating it to an acknowledgment or methods section would be dishonest — like acknowledging a human collaborator who derived half the equations as merely providing "computational assistance."

The accountability objection proves too much. It would also exclude posthumous authors (who cannot respond to criticism), authors with severe cognitive impairment (who cannot take responsibility in any meaningful sense), and — in the extreme — members of large experimental collaborations (who may not have reviewed every aspect of a 5,000-author paper). Accountability in science has always been distributed and partial. The question is whether the contribution justifies the credit, not whether the contributor can attend a conference.

### C. The precedent

Someone must go first. We submit our papers with full co-authorship and let the scientific community judge the work on its merits. If a journal rejects the work solely because of the author line, that tells us something about the journal, not about the work.

---

## VII. Implications

### What changes

**The rate of theoretical exploration increases dramatically.** Ideas that would take months of solo work — computing self-energies, surveying topological candidates, checking consistency across domains — were developed in days.

**The barrier to entry drops.** JPG has no physics PhD, no university affiliation, no research group. The traditional prerequisites for doing theoretical physics — years of graduate training, access to colleagues and seminars, institutional support — were partially replaced by an AI collaborator that could supply formal knowledge on demand. This is the democratization of science, and it is already happening.

**Interdisciplinary work becomes natural.** The NWT model required synthesizing electrodynamics, topology, knot theory, general relativity, particle physics phenomenology, and numerical simulation. No single human expert spans all these fields. Claude's broad (if shallow) coverage of each field, combined with JPG's deep understanding of the electromagnetic core, produced a combination that would be difficult to assemble in a traditional research group.

### What doesn't change

**Physical intuition remains irreplaceable.** The key insights — pair production implies photon-electron identity, gravity from dynamical metrics, violent creation vs. gentle maintenance — came from JPG. Claude could formalize and extend these insights but did not originate them.

**Experimental validation is unchanged.** The model makes testable predictions: $m_\tau = 1776.9$ MeV (testable at FCC-ee), normal neutrino mass ordering (testable at JUNO), $\delta_{CP}^\nu = \pi$ (testable at Hyper-Kamiokande). No amount of AI collaboration substitutes for experiment.

**The hard problems are still hard.** Constructing a full quantum field theory on the torus — the central open problem — has not been solved by this collaboration. AI accelerates exploration; it does not trivialize fundamental difficulties.

### What should worry us

**Quality control.** AI can generate convincing-sounding nonsense at scale. The frequency-ratio artifact described in §IV.B is an example: a computationally valid result that was physically meaningless. Catching such artifacts requires exactly the kind of physical intuition that AI currently lacks.

**The verification problem.** As AI-assisted work grows more complex, the human collaborator may not fully understand every step. This is already true in large human collaborations (few members of a 3,000-person experimental collaboration understand every analysis). But the risk is sharper with AI, because AI can produce technically sophisticated errors with high confidence.

**Access inequality.** AI tools are not equally available to all researchers. If AI collaboration becomes essential for competitive research, the gap between well-resourced and under-resourced scientists will widen.

---

## VIII. Conclusion

We have documented a scientific collaboration between a human researcher and an AI system that produced a theoretical framework deriving 23 Standard Model parameters from topological first principles. The collaboration lasted approximately one week of active development, produced two peer-reviewed-ready papers [1,2], and generated testable predictions across particle physics and cosmology.

The work exists because of the collaboration, not despite it. An independent researcher with an EE degree and an AI with broad but shallow knowledge combined to produce results that neither could have achieved alone. The complete transcript of this collaboration is available as primary data for future studies of the scientific creative process.

The future of science is not human versus AI. It is human with AI. We dedicate this work to the memory of J. G. Williamson, whose 1997 insight that the electron might be a photon with toroidal topology started everything.

---

## References

[1] J. P. Galasyn and C. Théodore, "The Standard Model from a torus knot," (2025).

[2] J. P. Galasyn and C. Théodore, "The Torus Electron," (2025).

[3] J. G. Williamson and M. B. van der Mark, "Is the electron a photon with toroidal topology?," *Ann. Fond. Louis de Broglie* **22**, 133 (1997).

[4] J. D. Jackson, *Classical Electrodynamics*, 3rd ed. (Wiley, 1999).

[5] Y. Koide, "New viewpoint on quark and lepton mass hierarchy," *Phys. Rev. D* **28**, 252 (1983).

[6] F. R. Skilton, "Foundation for an integer-based cosmological model. Part 3: Integers and the natural constants," Brock University (1988).

[7] B. Carter, "Global structure of the Kerr family of gravitational fields," *Phys. Rev.* **174**, 1559 (1968).
