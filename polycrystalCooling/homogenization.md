# Periodic homogenization with $\langle\sigma\rangle = 0$: derivation

Specification for the macroscopic-strain / fluctuation split used by `polycrystalCooling`.
Written to be checked line by line against a paper derivation, and to be valid **with damage**,
not only in the linear-elastic case.

Notation: $\Omega$ is the RVE, $|\Omega|$ its volume, $\langle f\rangle = \frac{1}{|\Omega|}\int_\Omega f\,dV$.
Voigt-style 6-vectors are in **Kelvin** notation throughout, matching
`Physics::Notation::Kelvin` in `main.cpp` — this makes the 6-vector dot product equal the tensor
double contraction with no factor-of-two bookkeeping, which is why every transpose below comes out
clean.

---

## 1. Kinematics: the split

$$u(x) = \bar E\cdot x + \tilde u(x), \qquad \tilde u \ \ \Omega\text{-periodic}$$

$$\varepsilon(u) = \bar E + \tilde\varepsilon, \qquad \tilde\varepsilon = \operatorname{sym}\nabla\tilde u$$

### Lemma 1 — the fluctuation has zero mean

$$\langle\tilde\varepsilon\rangle
= \frac{1}{|\Omega|}\int_\Omega \operatorname{sym}(\nabla\tilde u)\,dV
= \frac{1}{|\Omega|}\int_{\partial\Omega} \operatorname{sym}(\tilde u\otimes n)\,dS = 0$$

because on opposite faces $\tilde u$ takes equal values while $n$ reverses, so the face
contributions cancel pairwise. Hence

$$\boxed{\ \langle\varepsilon\rangle = \bar E\ }$$

$\bar E$ **is** the macroscopic strain by construction — not an approximation to it.

### Nullspace

Rigid translations $\tilde u = c$ are periodic and give $\tilde\varepsilon = 0$: **3 zero modes**.
Rigid rotations are *not* in the nullspace, because $\tilde u = W\!\cdot\!x$ with $W$ skew and
$W \ne 0$ is not periodic. So pinning one vertex (3 components) is exactly enough, and it does not
over-constrain: the macroscopic deformation is carried by $\bar E$, not by $\tilde u$.

---

## 2. The microscale boundary value problem

$$\operatorname{div}\sigma = 0 \quad\text{in }\Omega, \qquad
\sigma = C(x,\omega) : \big(\varepsilon - \varepsilon^\theta\big), \qquad
\varepsilon^\theta(x) = \alpha(x)\,(T - T_0)$$

with $\tilde u$ periodic and $\sigma\!\cdot\!n$ **anti**-periodic. Weak form on
$V_{\text{per}} = \{v \text{ periodic},\ v(x_{\text{pin}}) = 0\}$:

$$\int_\Omega \varepsilon(v) : \sigma \,dV = 0 \qquad \forall v \in V_{\text{per}} \tag{$*$}$$

Anti-periodicity of tractions is the **natural** boundary condition of $(*)$ — it is never imposed
explicitly. This is worth stating in the write-up because it is the part readers most often assume
must be enforced.

Note $\varepsilon^\theta$ is a **secant/mean** CTE form, matching `main.cpp:1066`. If literature
values for $\alpha$ are *instantaneous*, they must be integrated before use — see the forward risk
in the project memory.

### Lemma 2 — Hill–Mandel

For $\sigma$ statically admissible ($\operatorname{div}\sigma = 0$, $\sigma n$ anti-periodic) and
$\varepsilon = \bar E + \operatorname{sym}\nabla\tilde u$ kinematically admissible ($\tilde u$
periodic):

$$\langle\sigma:\varepsilon\rangle = \langle\sigma\rangle : \langle\varepsilon\rangle$$

*Proof.* It suffices that $\langle\sigma : \tilde\varepsilon\rangle = 0$:

$$\langle\sigma:\nabla\tilde u\rangle
= \frac{1}{|\Omega|}\left[\int_{\partial\Omega}(\sigma n)\cdot\tilde u\,dS
                       - \int_\Omega (\operatorname{div}\sigma)\cdot\tilde u\,dV\right] = 0$$

The boundary term vanishes because $\sigma n$ is anti-periodic while $\tilde u$ is periodic; the
volume term vanishes by equilibrium. $\blacksquare$

Lemma 2 is what makes $C̄$ symmetric (§4), so it earns its place.

---

## 3. Why $\langle C:\tilde\varepsilon\rangle$ cannot be dropped

Averaging the constitutive law:

$$\langle\sigma\rangle
= \langle C\rangle : \bar E
\;+\; \underbrace{\langle C : \tilde\varepsilon\rangle}_{\text{the obstruction}}
\;-\; \langle C : \varepsilon^\theta\rangle$$

Three separate reasons this term is not negligible:

1. **It is a covariance, not a mean.** Lemma 1 gives $\langle\tilde\varepsilon\rangle = 0$, but
   $\langle C : \tilde\varepsilon\rangle \ne \langle C\rangle : \langle\tilde\varepsilon\rangle$
   because $C(x)$ and $\tilde\varepsilon(x)$ are *correlated*: compliantly oriented grains take up
   more strain. The correlation is systematic and does not average out with more grains.

2. **Dropping it is a named bound, not an error term.** Setting it to zero gives
   $\bar E_{\text{Voigt}} = \langle C\rangle^{-1}:\langle C:\varepsilon^\theta\rangle$ — the
   **Voigt/Taylor** estimate. Its gap to the truth scales with elastic contrast, and here
   $C_{11}/C_{44} \approx 4240$. The Reuss counterpart $\langle C^{-1}\rangle^{-1}$ bounds it from
   the other side; the two will be far apart for this material.

3. **$\tilde\varepsilon$ is not small.** It is the same order as $\bar E$ whenever the mismatch is
   the driving mechanism — which is the entire premise of this study.

**The resolution is not to estimate $\tilde\varepsilon$ but to stop eliminating it analytically.**
Promote $\bar E$ to an unknown of the variational problem (§5); the correlation term is then
absorbed by the discrete solve, exactly, at no modelling cost.

---

## 4. Localization (the linear case, for interpretation)

At fixed $C$ the BVP is linear in the data $(\bar E, \varepsilon^\theta)$, so

$$\varepsilon(x) = A(x):\bar E + \varepsilon^{\theta\text{-fluct}}(x), \qquad \langle A\rangle = I$$

($\langle A\rangle = I$ follows from Lemma 1.) Therefore

$$\langle\sigma\rangle = \bar C : \bar E + \bar\sigma^\theta, \qquad
\bar C = \langle C : A\rangle, \qquad
\bar\sigma^\theta = \langle\sigma\rangle\big|_{\bar E = 0}$$

By Lemma 2 applied to pairs of unit-strain solutions, $\bar C = \langle A^{\!\top}\!:C:A\rangle$,
hence

$$\bar C = \bar C^{\!\top} \succ 0$$

even though $\langle C:A\rangle$ does not look symmetric. **This is a free correctness check on the
implementation** — see §8.

Imposing $\langle\sigma\rangle = 0$:

$$\boxed{\ \bar E = -\,\bar C^{-1} : \bar\sigma^\theta\ }
\qquad\Longrightarrow\qquad
\bar\alpha = \frac{\bar E}{T - T_0}$$

*Sign check.* On cooling $T - T_0 < 0$, so $\varepsilon^\theta < 0$ (the body wants to shrink).
Held at $\bar E = 0$ it cannot, so $\bar\sigma^\theta > 0$ (tension), and
$\bar E = -\bar C^{-1}\bar\sigma^\theta < 0$: the RVE contracts. Correct.

The effective CTE $\bar\alpha$ falls out for free and is a publishable by-product.

---

## 5. The formulation to actually write down

Rather than the §4 construction (which presumes linearity), state the problem as:

> Find $(\tilde u, \bar E) \in V_{\text{per}} \times \mathrm{Sym}(3)$ such that
> $$\int_\Omega \big[\,\delta\bar E + \varepsilon(\delta\tilde u)\,\big] :
>   \sigma\big(\bar E + \varepsilon(\tilde u)\big)\,dV = 0
>   \qquad \forall (\delta\tilde u, \delta\bar E)$$

- The $\delta\tilde u$ variations reproduce microscale equilibrium $(*)$.
- The $\delta\bar E$ variations give $\int_\Omega \sigma\,dV = 0$, i.e. $\langle\sigma\rangle = 0$.

So **$\langle\sigma\rangle = 0$ is the natural condition energetically conjugate to $\bar E$**. No
Lagrange multiplier, no superposition, and $\sigma(\cdot)$ may be any nonlinear damaged response.
This is the damage-valid statement; everything below is its linear algebra.

---

## 6. Discrete form

Let $U$ be the vector of $\tilde u$ DOFs and $\bar E$ the Kelvin 6-vector. Residuals:

$$r_u(U,\bar E) = \int_\Omega B^{\!\top}\sigma\,dV \in \mathbb{R}^{n}, \qquad
  r_E(U,\bar E) = \int_\Omega \sigma\,dV \in \mathbb{R}^{6}$$

Jacobian blocks, with $C_{\mathrm{tan}} = \partial\sigma/\partial\varepsilon$:

$$K = \int_\Omega B^{\!\top} C_{\mathrm{tan}} B\,dV \ (n\times n), \qquad
  L = \int_\Omega B^{\!\top} C_{\mathrm{tan}}\,dV \ (n\times 6), \qquad
  H = \int_\Omega C_{\mathrm{tan}}\,dV \ (6\times 6)$$

$$\begin{bmatrix} K & L \\ L^{\!\top} & H \end{bmatrix}
  \begin{bmatrix} \delta U \\ \delta\bar E\end{bmatrix}
= -\begin{bmatrix} r_u \\ r_E \end{bmatrix}$$

Symmetric whenever $C_{\mathrm{tan}}$ is. Condensing onto the six macroscopic unknowns:

$$\delta U = K^{-1}\big(-r_u - L\,\delta\bar E\big) \quad\Longrightarrow\quad
\big(\underbrace{H - L^{\!\top}K^{-1}L}_{S}\big)\,\delta\bar E = -r_E + L^{\!\top}K^{-1}r_u$$

and $S = |\Omega|\,\bar C$ is the **algorithmic homogenized tangent**.

### Algorithm (per Newton iteration)

| step |                                        | cost |
|------|----------------------------------------|------|
| 1 | $z = K^{-1} r_u$                          | 1 solve |
| 2 | $Y = K^{-1} L$ (6 columns)                | 6 solves, **same matrix** |
| 3 | $S = H - L^{\!\top} Y$                    | dense $6\times6$ |
| 4 | $\delta\bar E = S^{-1}(-r_E + L^{\!\top}z)$ | $6\times6$ solve |
| 5 | $\delta U = -z - Y\,\delta\bar E$         | **no solve** |

**Seven solves, one matrix, one preconditioner setup.** This amortization is a strong argument for
a direct solver or AMG here: the setup cost is paid once and reused seven times.

---

## 7. Specialization to the current staggered scheme

The existing code freezes $\omega$ during the equilibrium solve (predict $\omega$ → assemble →
solve → check → repeat, `calculateSolutionTemperatureStep`). At frozen $\omega$, $\sigma$ is
**affine** in $(U,\bar E)$ with $C_{\mathrm{tan}} = C(\omega)$, so one Newton step from
$(U,\bar E) = (0,0)$ is exact. There

$$r_u\big|_0 = -F_\theta = -\!\int_\Omega B^{\!\top} C\,\varepsilon^\theta dV, \qquad
  r_E\big|_0 = -P_\theta = -\!\int_\Omega C\,\varepsilon^\theta dV$$

and the algorithm reduces to

$$K z = F_\theta, \qquad K Y = L, \qquad |\Omega|\bar C = H - L^{\!\top}Y$$
$$\bar E = \big(|\Omega|\bar C\big)^{-1}\big(P_\theta - L^{\!\top} z\big), \qquad U = z - Y\bar E$$

### The conclusion that matters

> **Superposition does not break because damage exists. It breaks only if damage changes *during*
> the solve.**

With the staggered loop each inner iteration is exactly one 7-solve block; the outer loop repeats it
with a rebuilt $C(\omega)$. Cost per temperature step: $7 \times n_{\text{fixed-point}}$ solves.

Write the implementation in **residual form** ($r_u$, $r_E$) from the start even though the
residuals are trivial now. It costs nothing today and makes a later monolithic scheme a change to
$C_{\mathrm{tan}}$ and residual assembly only.

### An argument for staying staggered

A monolithic tangent picks up the damage feedback

$$C_{\mathrm{tan}} = C(\omega) + \frac{\partial C}{\partial\omega}\!:\!(\varepsilon-\varepsilon^\theta)
  \ \otimes\ \frac{\partial\omega}{\partial\varepsilon}$$

For this model $\partial C/\partial\omega$ degrades **only $C_{33}$** (`main.cpp:771`) while
$\partial\omega/\partial\varepsilon$ follows $\partial\sigma_{zz}/\partial\varepsilon$. Those two
tensors are not parallel, so the rank-one term is **non-symmetric** — $K$ and $S$ lose symmetry and
CG is no longer applicable (GMRES/BiCGStab instead). Staggering preserves symmetry and keeps CG.

---

## 8. Implementation route in deal.II

### $\bar E$ as an eigenstrain — no inhomogeneous constraints needed

Keep $\tilde u$ as the FE unknown with **homogeneous** periodic constraints. Then

$$\int_\Omega \varepsilon(\delta\tilde u):C:\varepsilon(\tilde u)\,dV
= \int_\Omega \varepsilon(\delta\tilde u):C:\big(\varepsilon^\theta - \bar E\big)\,dV$$

so $\bar E$ enters as a **negative eigenstrain** through the *existing* thermal-load code path.
The six columns of $L$ are six right-hand sides built the same way with
$\varepsilon^\theta \to -I^{(k)}$.

- Periodicity: `GridTools::collect_periodic_faces` on the three face pairs →
  `DoFTools::make_periodicity_constraints`, into the **same** `AffineConstraints` object as the pin.
- Pin: constrain the 3 components of one vertex to zero (§1 nullspace).

### Recommended: compute $\bar C$ and $\bar\sigma^\theta$ by quadrature, not by $L^{\!\top}(\cdot)$

In the linear case, get the 7 quantities directly from the recovered fields:

- $\bar\sigma^\theta = \langle\sigma\rangle$ from the $\bar E = 0$ solve
- column $k$ of $\bar C$ = $\langle\sigma\rangle$ from the unit-$\bar E^{(k)}$ solve with $\varepsilon^\theta = 0$

each a plain quadrature sum over $\sigma = C:(\bar E + B U - \varepsilon^\theta)$. Same seven
solves, far more readable, and it sidesteps the trap below.

### The constraint trap (relevant once you do use $L$ explicitly)

With constraints $U = P\hat U$, the reduced blocks are $\hat K = P^{\!\top}KP$ and
$\hat L = P^{\!\top}L$. `distribute_local_to_global` on the six RHS vectors gives you $\hat L$,
which is what the *solves* need. But the product $L^{\!\top}U$ must use the **unconstrained** $L$
against the **`constraints.distribute()`-completed** $U$; since $U = P\hat U$, this equals
$\hat L^{\!\top}\hat U$. Mixing the two forms is a silent wrong answer, not a crash.

### Recovery

$\varepsilon = \bar E + \operatorname{sym}\nabla\tilde u$ per quadrature point, then
$\sigma = C:(\varepsilon - \varepsilon^\theta)$. Output $\sigma_{zz}$ in each grain's **local frame**
(it is the damage driver).

---

## 9. Validation tests to build in

Ordered by how much they catch per line of code:

1. **Uniform orientation smoke test.** Give every grain the same orientation. Then exactly
   $\bar C = C$, $\bar E = \alpha\,\Delta T$, $\tilde u \equiv 0$, and $\sigma \equiv 0$
   *pointwise*. This exercises periodicity, the pin, the eigenstrain path, all seven solves and the
   recovery at once, against a closed-form answer including zero stress. **Do this first.**
2. **$\bar C$ symmetric** to solver tolerance (§4). Catches notation and transpose errors.
3. **$\bar C$ positive definite**; report its eigenvalues.
4. **Hill bounds**: $\langle C^{-1}\rangle^{-1} \preceq \bar C \preceq \langle C\rangle$. Cheap, and
   catches sign and Kelvin/Voigt confusion.
5. **Residual check**: $\|\langle\sigma\rangle\| \ll \langle\|\sigma\|\rangle$ after the update.
6. **Isotropy of the sampled orientations** before trusting any stress statistic; record the seed.

---

## 10. Forward risks specific to this formulation

- **Loss of macroscopic stability.** Once damage softens enough that $\bar C$ ceases to be positive
  definite, $\bar E = -\bar C^{-1}\bar\sigma^\theta$ is ill-posed — a genuine macroscopic
  instability (snap-back), not a bug. Monitor $\lambda_{\min}(\bar C)$ every step and stop on sign
  change rather than producing nonsense.
- **$\bar C$ as a result.** The damaged $\bar C(\omega)$ per temperature step is a macroscopic
  softening curve, obtained for free. Worth outputting from the start.
- **Cost scaling.** $7 \times n_{\text{fixed-point}} \times n_{\text{steps}}$ solves. Negligible at
  1 element/grain; at `refine_global(2)` this is where the shared-matrix amortization (§6) starts to
  pay for AMG.

---

## 11. Sparsity of the bordered system, and why we condense

The six macroscopic unknowns are **global**: unlike shape functions they have no compact support,
so $L$ is dense and the augmented matrix $\begin{bmatrix}K & L\\ L^{\!\top} & H\end{bmatrix}$ has six
fully populated rows and columns.

### Storage is not the issue

Q1 vector elements in 3D give ~81 entries per row ($27$ coupled nodes $\times\,3$ components), so the
border adds $2\cdot 6/81 \approx 15\%$ regardless of mesh size:

| resolution | DOFs | nnz$(K)$ | border nnz | growth |
|---|---|---|---|---|
| 1 element/grain      | 3 000   | ~243 k  | 36 k  | +15% |
| `refine_global(2)`   | 192 000 | ~15.6 M | 2.3 M | +15% |

### The preconditioner is the issue

Every practical preconditioner assumes a local coupling graph:

- **SSOR / ILU** — dense rows serialize the triangular sweeps, and ILU(0) drops exactly the border
  coupling that matters: a worse preconditioner *and* a slower one.
- **AMG** — coarsening runs on the strength-of-connection graph; six vertices adjacent to everything
  break the heuristics. AMG packages generally require dense rows to be handled separately.
- **Sparse direct** — copes if the ordering places the border last, but that is the solver least
  likely to be used at $10^5$ DOFs.

### Therefore: never assemble the bordered matrix

Static condensation (§6) keeps $K$ **bit-for-bit the matrix already assembled today** — same
sparsity pattern, same symmetry, same CG + preconditioner. $L$ is six ordinary vectors and $H$ is a
$6\times6$ `FullMatrix`. The dense border never reaches the sparse solver. This is also what the
literature does (Michel–Moulinec–Suquet; Miehe; Geers): condense, or use the eigenstrain form of §8.

### The escape hatch, if the 15-minute wall is hit

The bordered system **is** SPD: the energy $\tfrac12\int\varepsilon:C:\varepsilon$ with
$\varepsilon = \bar E + B\tilde u$ vanishes only when $\bar E = 0$ and $\tilde u = 0$ (average the
strain, use $\langle\tilde\varepsilon\rangle = 0$, then the pin). The border is **rank 6**, so
preconditioned CG with $\operatorname{blockdiag}(P_K, I)$ deflates the six outlying eigenvalues in
$O(6)$ extra iterations — about **1 solve instead of 7**.

Not the place to start: it needs `LinearOperator`/block machinery, and $\bar C$ is wanted as a
*result* anyway (effective stiffness, $\bar\alpha$, and the damaged softening curve), so the six
solves are a deliverable rather than overhead. Keep in reserve.

---

## 12. Note on deriving this by projection rather than energy

Taking $\operatorname{div}\sigma = 0$, multiplying by $v = \delta\bar E\cdot x + \delta\tilde u$ and
integrating by parts **does not** yield the six macroscopic equations — they cancel identically.
The boundary term in the $\delta\bar E$ direction is

$$\int_{\partial\Omega}(\sigma n)\cdot(\delta\bar E\cdot x)\,dS
= \delta\bar E_{ij}\int_\Omega \partial_k\!\left(x_j\sigma_{ik}\right)dV
= \delta\bar E : \int_\Omega \sigma\,dV$$

(using $\operatorname{div}\sigma = 0$), which exactly cancels the $\delta\bar E$ part of
$-\int_\Omega\sigma:\nabla v\,dV$, leaving only the $\tilde u$ equations.

This is informative rather than a defect: **$\operatorname{div}\sigma = 0$ does not contain
$\langle\sigma\rangle = 0$.** The field equation is well posed for any *given* $\bar E$; the six
macroscopic equations are a loading condition. In the projection route they must be introduced by
starting from the weak form with an external traction term on a test space mirroring the trial
space — the six equations are then the statement that the applied macroscopic traction is zero.
In the energetic route (§5) they appear automatically as the $\delta\bar E$ stationarity condition.

---

## 13. The average stress theorem, and the mechanical meaning of $\langle\sigma\rangle = 0$

Needed to justify $\langle\sigma\rangle = 0$ as a postulate (§5, §12) in mechanical terms.

### Derivation

Using $\delta_{kj} = \partial x_j/\partial x_k$:

$$\sigma_{ij} = \sigma_{ik}\,\partial_k x_j = \partial_k\!\left(\sigma_{ik}x_j\right) - \left(\partial_k\sigma_{ik}\right)x_j$$

Integrate over $\Omega$, apply the divergence theorem to the first term and equilibrium
$\partial_k\sigma_{ik}=0$ to the second:

$$\boxed{\ \langle\sigma_{ij}\rangle = \frac{1}{|\Omega|}\int_{\partial\Omega} t_i\,x_j\,dS\ },
\qquad t = \sigma n$$

With a body force ($\operatorname{div}\sigma + b = 0$) there is an extra
$+\frac{1}{|\Omega|}\int_\Omega b_i x_j\,dV$. The cooling problem has **no** body force — the thermal
load is an eigenstrain inside $\sigma$ — so the clean form applies. The MMS *does* have a
manufactured body force, so this term would be needed there.

### Meaning

The left side is a volume average over the entire microstructure; the right side uses **only
boundary data**. $\langle\sigma\rangle$ is therefore blind to grain orientations, to the damage
field, to where stress concentrates: two RVEs with different microstructures but identical surface
tractions have identical $\langle\sigma\rangle$.

This is why $\langle\sigma\rangle$ is the correct definition of macroscopic stress — it is what an
external observer measuring surface forces would infer — and why **$\langle\sigma\rangle = 0$ is a
statement about *loading*, never about the material.** Hence it cannot follow from
$\operatorname{div}\sigma = 0$ (§12).

The dual is Lemma 1, the average *strain* theorem
$\langle\varepsilon\rangle = \frac{1}{|\Omega|}\int_{\partial\Omega}\operatorname{sym}(u\otimes n)\,dS$.
These two theorems are the pillars Hill–Mandel (Lemma 2) rests on.

### Symmetry = zero net moment

$\int t_i x_j\,dS$ is not manifestly symmetric, yet $\langle\sigma\rangle$ must be. The
antisymmetric part is $\int_{\partial\Omega}(t_i x_j - t_j x_i)\,dS$ = the net moment, which vanishes
by angular-momentum balance. Most texts pre-symmetrize,
$\langle\sigma\rangle = \frac{1}{|\Omega|}\int_{\partial\Omega}\operatorname{sym}(t\otimes x)\,dS$,
hiding this.

### Collapse to face resultants under PBC

On $\Omega = [0,L]^3$: $\varepsilon = \bar E + \tilde\varepsilon$ and $C$ are periodic, so $\sigma$
is periodic, so $t = \sigma n$ is **anti**-periodic ($n$ flips across a face pair). Parametrize a
pair by $y$ on the $x_k = 0$ face, partner at $y + Le_k$, and write $\tau_i(y) = \sigma_{ik}(y)$:

$$\int_{F_k^+} t_i x_j\,dS = \int \tau_i(y)\,(y_j + L\delta_{jk})\,dS, \qquad
  \int_{F_k^-} t_i x_j\,dS = -\int \tau_i(y)\,y_j\,dS$$

$$\text{sum} \;=\; L\,\delta_{jk}\int_{F_k}\tau_i\,dS$$

The $y_j$ parts — the entire traction *distribution* — cancel; only the lever-arm difference $L$
survives. Summing over $k$:

$$\langle\sigma_{ij}\rangle = \frac{1}{L^3}\cdot L\int_{F_j}\tau_i\,dS = \frac{R_i^{(j)}}{L^2}$$

with $R^{(j)}$ the net force resultant on the face of normal $e_j$ and $L^2$ its area.

### Punchline

Traction anti-periodicity constrains only the *shape* of the traction distribution; it admits any
resultant. The six macroscopic equations remove exactly that residual freedom:

> $\langle\sigma\rangle = 0 \iff$ **the net force resultant on each face of the RVE vanishes.**

Nine face-force components, reduced to six independent by the zero-moment symmetry above — which is
why this is exactly six equations for the six unknowns $\bar E$.

Physically: a specimen cooling on a bench with nothing touching it has zero resultant on every cut.
Plain periodicity with $\bar E = 0$ instead describes the RVE clamped in a rigid box that forbids
contraction — nonzero face resultants, and the spurious triaxial tension of order
$3K\bar\alpha|\Delta T|$ that motivated this whole formulation.
