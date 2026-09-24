# A thermodynamic potential for the current model

Written so that future features can be added *inside* a potential framework rather than bolted on
and checked afterwards. The claim of this document is that **the model as implemented today is
already derivable from a Helmholtz free energy**, with one exception (the kinetic law), and that
writing the potential down tells you what each planned feature must look like.

Companion to `../polycrystalCooling/homogenization.md`, which covers the macroscopic/fluctuation
split. This one is purely local — it concerns the constitutive law at a point.

---

## 1. State variables and the potential

State: total strain $\varepsilon$, temperature $T$, scalar damage $\omega$, and the grain's c-axis
$n$ (fixed, not a state variable).

Write $\Delta T = T - T_0$ and let $N^{(2)} = n\otimes n$, $N^{(4)} = n\otimes n\otimes n\otimes n$.
The implemented material is

$$C(\omega,T) = C^{0}(T) - \omega\,\hat C_{33}(T)\,N^{(4)},
\qquad \alpha(\omega,T) = \alpha^{0}(T) - \omega\,\hat\alpha_{33}(T)\,N^{(2)}$$

with $\hat C_{33} = C_{33}^{0}f_C(T)$ and $\hat\alpha_{33} = \alpha_{33}^{0}f_\alpha(T)$ — i.e.
damage scales exactly the 33 entry of each, which is what `getElasticityTensor` and `getcteTensor`
do. The thermal (eigen) strain is $\varepsilon^\theta = \alpha(\omega,T)\,\Delta T$ and the elastic
strain $\varepsilon^{e} = \varepsilon - \varepsilon^\theta$.

**Proposed free energy per unit volume:**

$$\boxed{\ \psi(\varepsilon,T,\omega)
= \tfrac12\,\varepsilon^{e}:C(\omega,T):\varepsilon^{e}
\;+\; \psi_{d}(\omega)
\;+\; \psi_{\mathrm{th}}(T)\ }$$

$\psi_{\mathrm{th}}(T)$ is the purely caloric part (heat capacity); it never enters the mechanics
because $T$ is prescribed here, but it is needed for the entropy. $\psi_d(\omega)$ is energy
**stored** in the damage — see §6, where it turns out to be the natural home for the threshold.
Today's model has $\psi_d \equiv 0$.

---

## 2. State laws

$$\sigma = \frac{\partial\psi}{\partial\varepsilon} = C(\omega,T):\varepsilon^{e},
\qquad
s = -\frac{\partial\psi}{\partial T},
\qquad
Y = -\frac{\partial\psi}{\partial\omega}$$

The first **reproduces the implemented stress exactly** — `getLocalStressTensor` returns
`C(omega,T) * (localStrain - cte(omega,T)*(T-T0))`. So the elastic part of the model is already
potential-derived; nothing needs to change for that.

---

## 3. The damage driving force

Differentiating at fixed $\varepsilon, T$, using
$\partial C/\partial\omega = -\hat C_{33}N^{(4)}$ and
$\partial\varepsilon^\theta/\partial\omega = -\hat\alpha_{33}N^{(2)}\Delta T$:

$$\frac{\partial\psi}{\partial\omega}
= \tfrac12\,\varepsilon^{e}:\!\Big(\tfrac{\partial C}{\partial\omega}\Big)\!:\varepsilon^{e}
\;-\; \sigma:\frac{\partial\varepsilon^\theta}{\partial\omega}
\;+\;\psi_d'(\omega)$$

With $\varepsilon^{e}:N^{(4)}:\varepsilon^{e} = (\varepsilon^{e}_{33})^{2}$ and
$\sigma:N^{(2)} = \sigma_{33}$ (both in the grain frame):

$$\boxed{\ Y = \underbrace{\tfrac12\,\hat C_{33}\,(\varepsilon^{e}_{33})^{2}}_{\text{(a) stiffness}}
\;\underbrace{-\;\hat\alpha_{33}\,\Delta T\,\sigma_{33}}_{\text{(b) eigenstrain}}
\;-\;\psi_d'(\omega)\ }$$

Two competing contributions, and reading them is the whole point of the exercise:

- **(a)** is the elastic energy released by softening $C_{33}$. It is a square, so **always $\ge 0$**
  — it always pushes damage forward, in tension and compression alike.
- **(b)** is the work done by the change in eigenstrain. Its sign is that of $-\Delta T\,\sigma_{33}$:
  **positive on cooling under c-axis tension, negative on heating under the same stress.**

---

## 4. Dissipation, checked against the actual solution

The Clausius–Duhem inequality for this state set reduces to

$$\mathcal{D} = Y\,\dot\omega \;\ge\; 0$$

The model currently drives damage by $\langle\sigma_{33}-\sigma_{th}\rangle$, **not** by $Y$, so
consistency is not automatic and has to be verified. Evaluated cell by cell on
`bc-resolution-sweep-cte/periodic_64cpg`, final step ($\Delta T = -200$ K):

| | |
|---|---|
| cells where damage is active ($\sigma_{33} > \sigma_{th}$) | 3563 of 64000 |
| of those, cells with $Y > 0$ | **3563 — all of them** |
| $Y < 0$ anywhere in the RVE | 4 cells, all *below* threshold, so $\dot\omega = 0$ there |
| median term (b) / term (a) | **2.12** |

**The model is thermodynamically consistent on cooling.** Not by construction, but demonstrably.

Note the last row: **the $\alpha_{33}$ coupling contributes about twice as much driving force as the
$C_{33}$ coupling.** The feature added for Mrozowski physics turns out to dominate the energetics.

---

## 5. What the potential says about each planned feature

### Heating ($\Delta T > 0$)

Term (b) flips sign. Under c-axis tension on heating, $-\hat\alpha_{33}\Delta T\sigma_{33} < 0$, and
since term (b) is the larger of the two by ~2x at these stress levels, **$Y$ can go negative**.

$\mathcal{D} = Y\dot\omega \ge 0$ then *forbids* $\dot\omega > 0$ and *permits* $\dot\omega < 0$.

**So the thermodynamics independently predicts healing on heating** — the feature already planned on
Mrozowski grounds. That is a genuine consistency check between the physics argument and the
potential, arrived at from opposite directions.

It also gives the admissibility criterion for free: **healing is allowed exactly where $Y \le 0$,**
and that is a computable field, not a modelling choice.

### Partial healing

$\dot\omega < 0$ is admissible wherever $Y \le 0$. The natural potential-derived form is a
dissipation potential $\varphi^{*}(Y;\omega)$, convex with a minimum at $Y=0$, from which
$\dot\omega = \partial\varphi^{*}/\partial Y$; then $\mathcal{D}\ge 0$ holds identically, in both
directions, with no case analysis.

Irreversibility (cracks close but surfaces persist) is expressed by making $\varphi^{*}$
**asymmetric** in $Y$ — a smaller coefficient for $Y<0$ than for $Y>0$ — or by carrying a separate
non-decreasing $\omega_{\max}$ and letting only the reversible part heal. The potential does not
decide which; it does say that whichever you choose must keep $\varphi^{*}$ convex.

### Crack closure (unilateral response)

Your call to keep the *stiffness* tension/compression symmetric is consistent with a potential, but
then closure must enter elsewhere. The clean potential-level statement is to degrade only the
tensile part of the c-axis energy:

$$\psi = \tfrac12\,\varepsilon^{e}:C^{0}:\varepsilon^{e}
 - \tfrac12\,\omega\,\hat C_{33}\,\langle\varepsilon^{e}_{33}\rangle_{+}^{2} + \dots$$

which leaves $\psi$ once-differentiable (so $\sigma$ stays continuous) but not twice — the tangent
jumps at $\varepsilon^{e}_{33}=0$. Term (a) of $Y$ then vanishes in compression, exactly as a closed
crack should release no energy on further closing.

### The threshold, which is currently ad hoc

$\sigma_{th} = 200$ MPa is an activation condition bolted onto the kinetic law. In a potential
framework it is a **stored energy**: take $\psi_d(\omega) = Y_c\,\omega$, so

$$Y = \text{(a)} + \text{(b)} - Y_c$$

and the damage criterion $Y>0$ *is* the threshold, with $Y_c$ the specific energy of damage
creation — a Griffith-like quantity with units J/m³ that can be measured, rather than a stress that
must be fitted.

**Calibration hint:** the smallest $Y$ among currently active cells is $1.81\times10^{6}$ J/m³, so
$Y_c \approx 1.8\times10^{6}$ J/m³ reproduces today's behaviour at this state. (Only approximately —
the map between a $\sigma_{33}$ threshold and a $Y$ threshold is state-dependent, which is itself an
argument for the energetic form.)

---

## 6. The one thing that is NOT potential-derived

The kinetic law

$$\dot\omega = A\left(\frac{\langle\sigma_{33}-\sigma_{th}\rangle}{\sigma_{th}}\right)^{m}
\frac{1}{(1-\omega)^{m}}$$

is driven by $\sigma_{33}$, not by $Y$. Two ways forward:

1. **Keep it and verify.** Legitimate, and §4 shows it currently passes on cooling. But the
   verification must be redone for every new feature, and heating will break it.
2. **Re-express it in $Y$.** $\dot\omega = \partial\varphi^{*}/\partial Y$ with
   $\varphi^{*} \propto \langle Y\rangle^{m+1}/(m+1)$ gives $\mathcal{D}\ge0$ identically, and the
   threshold moves into $\psi_d$ as above.

Option 2 is what "thinking in the frame of a potential" buys: consistency stops being something to
check and becomes something guaranteed. The cost is a recalibration of $A$, $m$ and the threshold
against $Y$ rather than $\sigma_{33}$.

---

## 7. Entropy, and why the secant/instantaneous CTE question is unavoidable

$$s = -\frac{\partial\psi}{\partial T}
= -\tfrac12\,\varepsilon^{e}:\frac{\partial C}{\partial T}:\varepsilon^{e}
\;+\;\Big[\underbrace{\alpha + \frac{d\alpha}{dT}\Delta T}_{\textstyle =\ d\varepsilon^\theta/dT}\Big]:\sigma
\;-\;\psi_{\mathrm{th}}'(T)$$

The coefficient conjugate to stress in the entropy is $d\varepsilon^\theta/dT$ — the **instantaneous**
CTE — regardless of how $\varepsilon^\theta$ is parameterised. The code writes
$\varepsilon^\theta = \alpha(T)\Delta T$, a **secant** form, so its instantaneous CTE is
$\alpha + \alpha'\Delta T$, not $\alpha$.

Consequence: literature values quoted as instantaneous $\alpha_{\mathrm{inst}}(T)$ must be
integrated, $\varepsilon^\theta = \int_{T_0}^{T}\alpha_{\mathrm{inst}}\,dT'$, before being used as
the secant $\alpha$. Currently the `*_functionOfTemperature` entries are 1.0 so $\alpha' = 0$ and
the two coincide — the distinction bites the moment temperature dependence is restored.

---

## 8. An alternative structure worth knowing about

Making $\varepsilon^\theta$ depend on $\omega$ is a shortcut: physically, the c-axis expansion is
absorbed by crack *opening*, not by a material whose CTE changed. The cleaner structure carries an
explicit inelastic strain $\varepsilon^{d}$ (crack-opening strain) as a state variable:

$$\psi(\varepsilon,\varepsilon^{d},T,\omega)
= \tfrac12(\varepsilon-\varepsilon^{\theta}_{0}-\varepsilon^{d}):C(\omega,T):(\varepsilon-\varepsilon^{\theta}_{0}-\varepsilon^{d}) + \dots$$

with conjugate force $-\partial\psi/\partial\varepsilon^{d} = \sigma$ and its own evolution law. The
apparent CTE reduction then *emerges* rather than being prescribed, and crack closure becomes a
constraint $\varepsilon^{d}_{33}\ge 0$ — a unilateral condition on a kinematic variable, which is
much easier to state and to satisfy than a non-smooth energy.

That is a bigger change than the current model needs, but it is the structure to move to if
Mrozowski closure becomes central.
